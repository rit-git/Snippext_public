from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

from .dataset import *
from .model import PredictionModel
from .train_util import *


def tagging_criterion(pred_labeled, y_labeled):
    """The loss function for tagging task (with float tensor input)

    Args:
        pred_labeled (Tensor): the predicted float tensor
        y_labeled (Tensor): the groundtruth float tensor

    Returns:
        Tensor: the cross entropy loss with the 0'th dimention ignored
    """
    # cross-entropy, ignore the 0 class
    loss_x = torch.sum(-y_labeled[:, 1:] * pred_labeled[:, 1:].log(), -1).mean()
    return loss_x


def classifier_criterion(pred_labeled, y_labeled):
    """The loss function for classification task (with float tensor input)

    Args:
        pred_labeled (Tensor): the predicted float tensor
        y_labeled (Tensor): the groundtruth float tensor

    Returns:
        Tensor: the cross entropy loss
    """
    loss_x = torch.sum(-y_labeled * pred_labeled.log(), -1).mean()
    return loss_x


def mixmatch(model, batch, num_aug=2, alpha=0.4, alpha_aug=0.4, u_lambda=0.5):
    """Perform one iteration of MixMatchNL

    Args:
        model (PredictionModel): the model state
        batch (tuple): the input batch
        num_aug (int, Optional): the number of augmented examples in the batch
        alpha (float, Optional): the parameter for MixUp
        alpha_aug (float, Optional): the parameter for MixDA
        u_lambda (float, Optional): the parameter controlling
            the weight of unlabeled data

    Returns:
        Tensor: the loss (of 0-d)
    """
    _, x, _, _, mask, tags_for_labeled_dataset, _, taskname = batch
    taskname = taskname[0]

    # two batches of labeled and two batches of unlabeled
    batch_size = x.size()[0] // (num_aug + 3)

    tags_for_labeled_dataset = tags_for_labeled_dataset[:batch_size] #primele batch_size (labeled_batch) de la 0 la batch_size

    # the unlabeled half
    original_unlabeled = x[batch_size:2 * batch_size] #the unlabeled dataset without aug

    # augmented
    augmented_labeled_dataset = x[2 * batch_size:3 * batch_size] # augmented dataset

    # augmented unlabeled
    augmented_unlabeled_dataset = [] #unlabeled dataset augmented
    for uid in range(num_aug):
        augmented_unlabeled_dataset.append(x[(3 + uid) * batch_size:(4 + uid) * batch_size])

    # labeled + original unlabeled (???? seems to be labeled + augmented unlabeled)
    x = torch.cat((x[:batch_size], x[3 * batch_size:]))

    # label guessing
    model.eval()
    u_guesses = []
    u_aug_enc_list = []
    _, _, _, u_enc = model(original_unlabeled, tags_for_labeled_dataset,
                           task=taskname, get_enc=True)

    for x_u in augmented_unlabeled_dataset:
        if alpha_aug <= 0:
            u_aug_lam = 1.0
        else:
            u_aug_lam = np.random.beta(alpha_aug, alpha_aug)

        # it is fine to switch the order of x_u and original_unlabeled in this case
        u_logits, tags_for_labeled_dataset, _, u_aug_enc = model(x_u, tags_for_labeled_dataset,
                                          augment_batch=(original_unlabeled, u_aug_lam),
                                          aug_enc=u_enc,
                                          task=taskname,
                                          get_enc=True)
        # softmax
        u_guess = F.softmax(u_logits, dim=-1)
        u_guess = u_guess.detach()
        u_guesses.append(u_guess)

        # save u_aug_enc
        u_aug_enc_list.append(u_aug_enc)

    # averaging
    u_guess = sum(u_guesses) / len(u_guesses)

    # temperature sharpening
    T = 0.5
    u_power = u_guess.pow(1 / T)
    u_guess = u_power / u_power.sum(dim=-1, keepdim=True)

    # make duplicate of u_guess
    if len(u_guess.size()) == 2:
        u_guess = u_guess.repeat(num_aug, 1)
    else:
        u_guess = u_guess.repeat(num_aug, 1, 1)

    vocab = u_guess.shape[-1]
    # switch back to training mode
    model.train()

    # shuffle
    index = torch.randperm(batch_size + u_guess.size()[0])
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)

    # convert y to one-hot
    y_onehot = F.one_hot(tags_for_labeled_dataset, vocab).float()
    y_concat = torch.cat((y_onehot, u_guess))
    y_mixed = y_concat[index, :]

    # x_aug_enc
    _, _, _, x_enc = model(x[:batch_size], tags_for_labeled_dataset,
                           task=taskname,
                           get_enc=True)
    # concatenate the augmented encodings
    x_enc = torch.cat([x_enc] + u_aug_enc_list)

    # forward
    if alpha_aug <= 0:
        aug_lam = 1.0
    else:
        aug_lam = np.random.beta(alpha_aug, alpha_aug)
    logits, y_concat, _ = model(x, y_concat,
                                augment_batch=(augmented_labeled_dataset, aug_lam),
                                x_enc=x_enc,
                                second_batch=(index, lam),
                                task=taskname)
    logits = F.softmax(logits, dim=-1)
    l_pred = logits[:batch_size].view(-1, vocab)
    u_pred = logits[batch_size:].view(-1, vocab)

    # mixup y's
    tags_for_labeled_dataset = lam * y_concat + (1.0 - lam) * y_mixed
    l_y = tags_for_labeled_dataset[:batch_size].view(-1, vocab)
    u_y = tags_for_labeled_dataset[batch_size:].view(-1, vocab)

    # cross entropy on label data + mse on unlabeled data
    if 'tagging' in taskname:
        loss_x = tagging_criterion(l_pred, l_y)
        loss_u = F.mse_loss(u_pred[:, 1:], u_y[:, 1:])
    else:
        loss_x = classifier_criterion(l_pred, l_y)
        loss_u = F.mse_loss(u_pred, u_y)

    loss = loss_x + loss_u * u_lambda
    return loss


# global bookkeeping variables for using the unlabeled set
epoch_idx = 0
u_order = []


def create_mixmatch_batches(labeled_set, labeled_augmented_set, unlabeled_set, unlabeled_augmented_set,
                            num_augmentations=2,
                            batch_size=16):
    """Create batches for mixmatchnl

    Each batch is the concatenation of (1) a labeled batch, (2) an augmented
    labeled batch (having the same order of (1) ), (3) an unlabeled batch,
    and (4) multiple augmented unlabeled batches of the same order
    of (3).

    Args:
        labeled_set (SnippextDataset): the train set
        labeled_augmented_set (SnippextDataset): the augmented train set
        unlabeled_set (SnippextDataset): the unlabeled set
        unlabeled_augmented_set (SnippextDataset): the augmented unlabeled set
        num_augmentations (int, optional): number of unlabeled augmentations to be created
        batch_size (int, optional): batch size (of each component)

    Returns:
        list of list: the created batches
    """
    print("Creating mixmatch batches")

    mixed_batches = []
    num_labeled = len(labeled_set)
    l_index = np.random.permutation(num_labeled) # l_index este un array de indecsi random de la 0 la num_labeled
    # num_unlabeled = len(u_set)
    # u_index = np.random.permutation(num_unlabeled)

    global u_order # lista de indecsi cu dim 0 -> len(u_set)
    if len(u_order) == 0:
        u_order = list(range(len(unlabeled_set)))
        random.shuffle(u_order)
        u_order = np.array(u_order)

    global epoch_idx
    u_index = np.random.permutation(num_labeled) + num_labeled * epoch_idx
    u_index %= len(unlabeled_set)
    u_index = u_order[u_index]
    epoch_idx += 1

    l_batch = []
    l_batch_aug = []
    u_batch = []
    u_batch_aug = [[] for _ in range(num_augmentations)]
    padder = labeled_set.pad

    for i, idx in enumerate(l_index):
        u_idx = u_index[i]
        l_batch.append(labeled_set[idx])
        # print("Labeled entry ", labeled_set[idx])
        l_batch_aug.append(labeled_augmented_set[idx])
        # print("Augmented entry", labeled_augmented_set[idx])

        # add augmented examples of unlabeled
        u_batch.append(unlabeled_set[u_idx])
        # print("Unlabeled entry ", unlabeled_set[u_idx])
        for uid in range(num_augmentations):
            u_batch_aug[uid].append(unlabeled_augmented_set[u_idx])
            # print("Unlabeled aug entry ", unlabeled_augmented_set[u_idx])
        if len(l_batch) == batch_size or i == len(l_index) - 1:
            batches = l_batch + u_batch + l_batch_aug
            for ub in u_batch_aug:
                batches += ub

            mixed_batches.append(padder(batches))
            l_batch.clear()
            l_batch_aug.clear()
            u_batch.clear()
            for ub in u_batch_aug:
                ub.clear()
    random.shuffle(mixed_batches)

    return mixed_batches


def train(model, l_set, aug_set, u_set, u_set_aug, optimizer,
          scheduler=None,
          batch_size=32,
          num_aug=2,
          alpha=0.4,
          alpha_aug=0.8,
          u_lambda=1.0):
    """Perform one epoch of MixMatchNL

    Args:
        model (PredictionModel): the model state
        train_dataset (SnippextDataset): the train set
        augment_dataset (SnippextDataset): the augmented train set
        u_dataset (SnippextDataset): the unlabeled set
        u_dataset_aug (SnippextDataset): the augmented unlabeled set
        optimizer (Optimizer): Adam
        scheduler (Scheduler, optional): the learning rate scheduler
        num_aug (int, Optional):
        batch_size (int, Optional): batch size
        alpha (float, Optional): the alpha for MixUp
        alpha_aug (float, Optional): the alpha for MixDA
        u_lambda (float, Optional): the weight of unlabeled data

    Returns:
        None
    """
    mixed_batches = create_mixmatch_batches(l_set,
                                            aug_set,
                                            u_set,
                                            u_set_aug,
                                            num_augmentations=num_aug,
                                            batch_size=batch_size // 2)
    print("Starting to train the model...")
    model.train()
    for i, batch in enumerate(mixed_batches):
        words, x, is_heads, tags, mask, y, seqlens, taskname = batch
        taskname = taskname[0]
        _y = y

        # perform mixmatch
        optimizer.zero_grad()
        try:
            loss = mixmatch(model, batch, num_aug, alpha, alpha_aug, u_lambda)
            loss.backward()

            optimizer.step()
            if scheduler:
                scheduler.step()

            if i == 0:
                print("=====sanity check======")
                print("words:", words[0])
                print("x:", x.cpu().numpy()[0][:seqlens[0]])
                print("tokens:", get_tokenizer().convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
                print("is_heads:", is_heads[0])
                y_sample = _y.cpu().numpy()[0]
                if np.isscalar(y_sample):
                    print("y:", y_sample)
                else:
                    print("y:", y_sample[:seqlens[0]])
                print("tags:", tags[0])
                print("mask:", mask[0])
                print("seqlen:", seqlens[0])
                print("task_name:", taskname)
                print("=======================")

            if i % 10 == 0:  # monitoring
                print(f"step: {i}, task: {taskname}, loss: {loss.item()}")
                del loss
        except:
            print("debug - seqlen:", max(seqlens))
            torch.cuda.empty_cache()


def initialize_and_train(task_config,
                         trainset,
                         augmentset,
                         validset,
                         testset,
                         uset,
                         uset_aug,
                         hp,
                         run_tag):
    """The train process.

    Args:
        task_config (dictionary): the configuration of the task
        trainset (SnippextDataset): the training set
        augmentset (SnippextDataset): the augmented training set
        validset (SnippextDataset): the validation set
        testset (SnippextDataset): the testset
        uset (SnippextDataset): the unlabeled dataset
        uset_aug (SnippextDataset): the unlabeled dataset, augmented
        hp (Namespace): the parsed hyperparameters
        run_tag (string): the tag of the run (for logging purpose)

    Returns:
        None
    """
    print("Initialize and train task_config", task_config, " hp ", hp, " run_tag ", run_tag)
    padder = SnippextDataset.pad

    # iterators for dev/test set
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 4,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 4,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PredictionModel([task_config], device, hp.finetuning, task_type=task_config['task_type'])
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # learning rate scheduler
    num_steps = (len(trainset) // hp.batch_size * 2) * hp.n_epochs
    print("Num steps ", num_steps)

    lambdaR = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_steps // 10,
                                                num_training_steps=num_steps)
    # create logging directory
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    writer = SummaryWriter(log_dir=hp.logdir)

    # start training
    best_dev_f1 = best_test_f1 = 0.0

    for epoch in range(1, hp.n_epochs + 1):
        print("Training epoch ", epoch)
        train(model,
              trainset,
              augmentset,
              uset,
              uset_aug,
              optimizer,
              scheduler=lambdaR,
              batch_size=hp.batch_size,
              num_aug=hp.num_aug,
              alpha=hp.alpha,
              alpha_aug=hp.alpha_aug,
              u_lambda=hp.u_lambda)

        print(f"=========eval at epoch={epoch}=========")
        dev_f1, test_f1 = eval_on_task(epoch,
                                       model,
                                       task_config['name'],
                                       valid_iter,
                                       validset,
                                       test_iter,
                                       testset,
                                       run_tag)

        if hp.save_model:
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                torch.save(model.state_dict(), run_tag + '_dev.pt')
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                torch.save(model.state_dict(), run_tag + '_test.pt')
        writer.close()