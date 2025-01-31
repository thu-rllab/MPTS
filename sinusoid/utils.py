import torch



def calcu_cvar(data, cvar_alpha):
    max_data = sorted(data, reverse=True)[0:int(len(data) * (1-cvar_alpha))]
    # loss, reverse=True;  Acc, reverse=False
    cvar=np.mean(max_data)
    return cvar

def log_string(file_out, out_str, print_out=True):
    file_out.write(out_str+'\n')
    file_out.flush()
    if print_out:
        print(out_str)

def pearson_correlation_coefficient_torch(x, y):
    """
    Compute the Pearson Correlation Coefficient using PyTorch.

    Args:
    x (torch.Tensor): First dataset.
    y (torch.Tensor): Second dataset.

    Returns:
    torch.Tensor: Pearson Correlation Coefficient.
    """
    # Ensure inputs are torch tensors
    x = torch.tensor(x.clone().detach(), dtype=torch.float32)
    y = torch.tensor(y.clone().detach(), dtype=torch.float32)

    # Subtract means
    x_diff = x - torch.mean(x)
    y_diff = y - torch.mean(y)

    # Calculate numerator (covariance)
    numerator = torch.sum(x_diff * y_diff)

    # Calculate denominator (product of standard deviations)
    denominator = torch.sqrt(torch.sum(x_diff ** 2) * torch.sum(y_diff ** 2))

    # Prevent division by zero
    if denominator == 0:
        return torch.tensor(0.0)  # Return tensor(0.0) if no variation

    # Pearson correlation coefficient
    pcc = numerator / denominator

    return pcc



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation





def plot_3d_data(observed_data, draw_data, draw_data_grid, save_path, test_surface, predict_surface):


    scatter_x = observed_data[0][:,0].cpu()
    scatter_y = observed_data[0][:,1].cpu()
    scatter_z_gt = observed_data[1].cpu() #0.95

    scatter_z_test = draw_data[1].cpu() #0.93
    scatter_z_pred = draw_data[2].cpu()
    scatter_z_std = draw_data[3].cpu()

    grid_x = draw_data_grid[0][:,0].cpu()
    grid_y = draw_data_grid[0][:,1].cpu()
    grid_z_test = draw_data_grid[1].cpu()
    grid_z_pred = draw_data_grid[2].cpu()
    grid_z_std = draw_data_grid[3].cpu()



    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ++++++++
    grid_matrix_x, grid_matrix_y = np.mgrid[0:5:5 / 100, 0:np.pi:np.pi / 100]

    if test_surface == "observe_test":
        grid_x_extend_test, grid_y_extend_test, grid_z_pred_extend_test = scatter_x, scatter_y, scatter_z_test
        ax.scatter(scatter_x, scatter_y, scatter_z_test, color="black", edgecolors="black", marker="o", s=1,label='observed_data')

    elif test_surface == "observe_gt":
        grid_x_extend_test, grid_y_extend_test, grid_z_pred_extend_test = scatter_x, scatter_y, scatter_z_gt
        ax.scatter(scatter_x, scatter_y, scatter_z_gt, color="blue", edgecolors="blue", marker="o", s=1, label='observed_data')

    elif test_surface == "grid_test":
        grid_x_extend_test, grid_y_extend_test, grid_z_pred_extend_test = grid_x, grid_y, grid_z_test
        ax.scatter(scatter_x, scatter_y, scatter_z_test, color="black", edgecolors="black", marker="o", s=1,label='observation')

    elif test_surface == "observe_grid_test":
        index = torch.randperm(len(grid_x))[:10]
        grid_x_extend_test, grid_y_extend_test, grid_z_pred_extend_test = torch.cat([scatter_x, grid_x[index]]), torch.cat([scatter_y, grid_y[index]]), torch.cat([scatter_z_test, grid_z_test[index]])
        ax.scatter(scatter_x, scatter_y, scatter_z_gt, color="black", edgecolors="black", marker="o", s=1, label='observed_data')

    grid_matrix_z1 = griddata((grid_x_extend_test, grid_y_extend_test), grid_z_pred_extend_test, (grid_matrix_x, grid_matrix_y), method='cubic')
    surface1 = ax.plot_surface(grid_matrix_x, grid_matrix_y, grid_matrix_z1, cmap='viridis', alpha=0.8, label='test_LossSurface')

    # ++++++++
    if predict_surface == "observe_grid":
        index = torch.randperm(len(grid_x))[:100]
        grid_x_extend, grid_y_extend, grid_z_pred_extend, grid_z_std_extend = torch.cat([scatter_x, grid_x[index]]), torch.cat([scatter_y, grid_y[index]]), torch.cat([scatter_z_pred, grid_z_pred[index]]), torch.cat([scatter_z_std, 2*grid_z_std[index]])
        pdb.set_trace()
        ax.scatter(scatter_x, scatter_y, scatter_z_pred, color="red", edgecolors="red", marker="o", s=1, label='prediction')
        for i in range(len(scatter_x)):
            ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], scatter_y[i]], [scatter_z_gt[i], scatter_z_pred[i]], color='black', linestyle='--', linewidth=0.5)

    if predict_surface == "observe":
        grid_x_extend, grid_y_extend, grid_z_pred_extend, grid_z_std_extend = scatter_x, scatter_y, scatter_z_pred, scatter_z_std
        ax.scatter(scatter_x, scatter_y, scatter_z_pred, color="red", edgecolors="red", marker="o", s=1, label='prediction')
        for i in range(len(scatter_x)):
            ax.plot([scatter_x[i], scatter_x[i]], [scatter_y[i], scatter_y[i]], [scatter_z_gt[i], scatter_z_pred[i]], color='black', linestyle='--', linewidth=0.5)
    if predict_surface == "grid":
        grid_x_extend, grid_y_extend, grid_z_pred_extend, grid_z_std_extend = grid_x, grid_y, grid_z_pred, grid_z_std

    grid_z2 = griddata((grid_x_extend, grid_y_extend), grid_z_pred_extend, (grid_matrix_x, grid_matrix_y), method='cubic')
    grid_z2_1 = griddata((grid_x_extend, grid_y_extend), grid_z_pred_extend + 3*grid_z_std_extend, (grid_matrix_x, grid_matrix_y), method='cubic')
    grid_z2_2 = griddata((grid_x_extend, grid_y_extend), grid_z_pred_extend - 3*grid_z_std_extend, (grid_matrix_x, grid_matrix_y), method='cubic')

    surface2 = ax.plot_surface(grid_matrix_x, grid_matrix_y, grid_z2, cmap='magma', alpha=0.8, label='predicted_LossSurface')
    surface2_1 = ax.plot_surface(grid_matrix_x, grid_matrix_y, grid_z2_1, cmap='magma', alpha=0.2, label='predicted_LossSurface')
    surface2_2 = ax.plot_surface(grid_matrix_x, grid_matrix_y, grid_z2_2, cmap='magma', alpha=0.2, label='predicted_LossSurface')

    # Labels
    ax.set_xlabel('amp')
    ax.set_ylabel('phase')
    ax.set_zlabel('Loss')

    from matplotlib.patches import Patch
    legend1 = Patch(facecolor='blue', edgecolor='blue', label='observation')
    legend2 = Patch(facecolor='red', edgecolor='red', label='prediction')

    ax.legend(handles=[legend1, legend2], loc='upper right')
    # fig.colorbar(surface1, ax=ax, shrink=0.5, aspect=5)
    # fig.colorbar(surface2, ax=ax, shrink=0.5, aspect=5)
    # Color bar (optional)
    # fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)  # Adds a color bar to your plot
    plt.title("Loss surface on test tasks")
    ax.view_init(elev=10, azim=10)
    # plt.show()

    def update_view(angle):
        ax.view_init(elev=10, azim=angle)
        return fig

    ani = FuncAnimation(fig, update_view, frames=np.arange(0, 360, 5), blit=False)
    ani.save(save_path + "test_surface=" + test_surface + "_**_predict_surface=" + predict_surface + ".gif", fps=400, dpi=400)

    fig.savefig(save_path+"test_surface="+test_surface+"_**_predict_surface="+predict_surface+".png", dpi=400)
    print(save_path+"test_surface="+test_surface+"_**_predict_surface="+predict_surface+".png"+ "  The image is ready!")



def plot_3d_data_train(observed_data, x, y, test_mse, predicted_mean, predicted_variance, save_path=None):

    # 0.1, 5, 0, np.pi
    # Create a grid to interpolate onto
      # 100j specifies 100 points in the complex number notation

    scatter_x = observed_data[0][:,0].cpu()
    scatter_y = observed_data[0][:,1].cpu()
    scatter_z = observed_data[1].cpu()

    x = x.reshape(100, 100)
    y = y.reshape(100, 100)
    z1 = test_mse.reshape(100, 100)
    z2 = predicted_mean.reshape(100, 100)
    predicted_variance = predicted_variance.reshape(100, 100)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')



    surface1 = ax.plot_surface(x, y, z1, cmap='viridis', alpha=0.7, label='test_LossSurface')

    surface2 = ax.plot_surface(x, y, z2, cmap='magma', alpha=0.7, label='predicted_LossSurface')
    # surface2_1 = ax.plot_surface(x, y, z2+predicted_variance, cmap='magma', alpha=0.2)
    # surface2_2 = ax.plot_surface(x, y, z2-predicted_variance, cmap='magma', alpha=0.2)

    ax.scatter(scatter_x, scatter_y, scatter_z, color="black", s = 0.1, label='observed_data')

    # ax.contour(x, y, z2, zdir='z', offset=-1, cmap='coolwarm')
    # ax.contour(x, y, z2, zdir='x', offset=-1, cmap='coolwarm')
    # ax.contour(x, y, z2, zdir='y', offset=0, cmap='coolwarm')

    # Labels
    ax.set_xlabel('amp')
    ax.set_ylabel('phase')
    ax.set_zlabel('Loss')
    fig.colorbar(surface1, ax=ax, shrink=0.5, aspect=5)
    fig.colorbar(surface2, ax=ax, shrink=0.5, aspect=5)
    # Color bar (optional)
    # fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)  # Adds a color bar to your plot

    plt.title("Loss surface on test tasks")

    ax.view_init(elev=0, azim=0)
    plt.show()

    fig.savefig(save_path, dpi=600)
    print(save_path+ "  The image is ready!")
