
import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    train_start_time = time.time() # timer for training
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers (if any)
    model.train()
    
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs_keep + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += 1
            epoch_iter += 1
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:    # print & plot training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        if epoch % opt.display_epoch_freq == 0:   # display images on tensorboard
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, total iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        model.update_learning_rate()    # update learning rates at the end of every epoch.

        message = 'End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs_keep + opt.n_epochs_decay, time.time() - epoch_start_time)
        print(message)
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
    
    train_end_message = 'End of training \t Time Taken: %.3f hours' % ((time.time() - train_start_time)/3600.0)
    print(train_end_message)
    with open(visualizer.log_name, "a") as log_file:
        log_file.write('%s\n' % train_end_message)  # save the message
