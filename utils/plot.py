import matplotlib.pyplot as plt
import matplotlib.patches as pat
import mpl_toolkits.axes_grid1

def single_plot(im, vmax, vmin, cmap, figsize=(10,5), title=None, save_path=None, is_colorbar=True):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if title != None:
        plt.title(title)
    
    ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    im = ax.imshow(im, vmax=vmax, vmin=vmin, cmap=cmap)
    if is_colorbar is True:
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        fig.colorbar(im, cax=cax)
    
    fig.show()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        print('saved to:', save_path)
    
    
    
    # ims.shape = (x, y, num)
def multiple_plot(ims, vmax, vmin, cmap, row_num=2, figsize=(15,10), title=None, save_path=None):
    num = ims.shape[2]

    fig = plt.figure(figsize=figsize)
    for i in range(0, num):
        ax = fig.add_subplot(row_num, int(num/row_num), i+1)
        if title != None:
            plt.title(title)
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
        ax.tick_params(bottom=False,
                        left=False,
                        right=False,
                        top=False)
        im = ax.imshow(ims[:, :, i], vmax=vmax, vmin=vmin, cmap=cmap)
        fig.colorbar(im, cax=cax)
        
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        print('saved to:', save_path)
        
def pred_plot(drs_pred, args, figsize=(12, 4), save_path=None):
    assert len(drs_pred.shape) == 3
    fig = plt.figure(figsize=figsize)
    
    ax = fig.add_subplot(131)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(drs_pred[:, :, 0], 
                   vmax=args.dist_max * args.scene_scale, 
                   vmin=args.dist_min * args.scene_scale, cmap='jet')
    ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    fig.colorbar(im, cax=cax)
    
    ax = fig.add_subplot(132)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(drs_pred[:, :, 1], 
                   vmax=args.ref_max, 
                   vmin=args.ref_min, cmap='gray')
    ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    fig.colorbar(im, cax=cax)
    
    ax = fig.add_subplot(133)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    im = ax.imshow(drs_pred[:, :, 2], 
                   vmax=args.sigma_max, 
                   vmin=args.sigma_min, cmap='jet')
    ax.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)
    ax.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    fig.colorbar(im, cax=cax)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        print('saved to:', save_path)
    
    plt.show()