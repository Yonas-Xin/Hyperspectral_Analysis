import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gdal_utils import colors

def show_scatter(data_pca, mask):
    mask = mask[mask != 0]
    label = np.unique(mask)
    dataset_all = []
    legend = []
    for i in label:
        legend.append(f'Class{i}')
        data = data_pca[mask==i]
        dataset_all.append(data)
    plt.figure(1,(16,12))
    for i in range(len(dataset_all)):
        x = dataset_all[i]
        plt.scatter3D(xs=x[:, 0], ys=x[:, 1],zs=x[:,2], color=colors[i])
    plt.legend(legend, loc='upper right')
    plt.show()


def get_ellipsoid(center, radii, rotation, n=20):
    """生成椭球面点"""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # 应用旋转和平移
    for i in range(n):
        for j in range(n):
            [x[i, j], y[i, j], z[i, j]] = np.dot(rotation, [x[i, j], y[i, j], z[i, j]]) + center
    return x, y, z
def create_scatter_3d(data_pca, mask,  show_connections=True, show_ellipsoid=True,
                      x_range=None, y_range=None, z_range=None,show_legend=True):
    '''
    :param data_pca:
    :param mask: 从1开始记录样本位置
    :param show_connections:
    :return:'''
    label = np.unique(mask)
    dataset_all = []
    legend = []
    # mean = np.empty((len(label),3),dtype=np.float32)
    for i in label:
        if i <= 0:
            continue
        legend.append(f'Class{i}')
        data = data_pca[mask == i]
        dataset_all.append(data)

    fig = go.Figure(layout={"autosize": True})
    for i in range(len(dataset_all)):
        x = dataset_all[i]
        if x.shape[0] > 300 and len(dataset_all)>1:
            indices = np.random.choice(x.shape[0], size=300, replace=False)
            scatter_point = x[indices]
        else:
            scatter_point = x
        point_mean = np.mean(x, axis=0, keepdims=True)
        # x = np.concatenate((x, point_mean), axis=0)
        # 添加三维散点轨迹
        fig.add_trace(
            go.Scatter3d(
                x=scatter_point[:, 0],
                y=scatter_point[:, 1],
                z=scatter_point[:, 2],
                mode='markers',
                name=f'Class {i + 1}',
                marker=dict(
                    size=3,  # 优化点大小
                    color=colors[i],  # 指定颜色
                    opacity=0.9,  # 增加透明度
                    line=dict(width=0.1)  # 添加点边界线
                ),
                showlegend=show_legend  # 控制是否显示图例
            )
        )
        # 添加连接线
        if show_connections:
            # 生成连线坐标（包含None分隔符）
            line_x, line_y, line_z = [], [], []
            for point in range(x.shape[0]):
                line_x.extend([x[point][0], point_mean[0, 0], None])  # 数据点 -> 中心点 -> None
                line_y.extend([x[point][1], point_mean[0, 1], None])
                line_z.extend([x[point][2], point_mean[0, 2], None])

            fig.add_trace(
                go.Scatter3d(
                    x=line_x, y=line_y, z=line_z,
                    mode='lines',
                    name=f'lines {i+1}',
                    line=dict(width=1, color=colors[i]),
                    showlegend=False,  # 不在图例中显示连线
                    hoverinfo='skip'  # 禁用悬停信息
                )
            )
        if show_ellipsoid:
            # 计算协方差矩阵
            cov_matrix = np.cov(x, rowvar=False)
            # 计算椭球的参数（特征值和特征向量）
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            radii = np.sqrt(eigenvalues) * 2.5  # 2倍标准差作为椭球大小
            # 生成椭球表面
            ellipsoid = get_ellipsoid(center=point_mean[0], radii=radii, rotation=eigenvectors, n=100)
            # 添加椭球表面
            fig.add_trace(
                go.Surface(
                    x=ellipsoid[0],
                    y=ellipsoid[1],
                    z=ellipsoid[2],
                    opacity=0.3,
                    surfacecolor=np.zeros_like(ellipsoid[0]),  # 需要一个与 z 维度匹配的数组
                    colorscale=[[0, colors[i]], [1, colors[i]]],  # 让整个曲面使用一种颜色
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range),  # 设置x轴范围
            yaxis=dict(range=y_range),  # 设置y轴范围
            zaxis=dict(range=z_range),  # 设置z轴范围
            camera=dict(eye=dict(x=2, y=-1, z=0.6)),  # 调整默认视角
            aspectratio = dict(x=1, y=1, z=1))  # 设置 x:y:z 的比例
        )
    return fig
def show_scatter3d(fig, show_axis=True,show_title=True, show_legend=True):
    # 图表布局优化 =========================================
    if show_title:
        fig.update_layout(title={
            'text': "3D PCA Visualization",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        })
    if show_axis:
        fig.update_layout(scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'))
    if show_legend:
        fig.update_layout(
        legend=dict(
            x=0.75,  # 图例位置调整
            y=0.95,
            bgcolor='rgba(255,255,255,1)')
        )
    fig.show()

def multiplot_scatter_3d(rows, cols, figs:list, title='3D PCA Visualization'):
    specs_list = [[{'type': 'scatter3d'}]*cols]*rows
    fig = make_subplots(rows, cols, specs=specs_list)
    for i in range(len(figs)):
        for trace in figs[i].data:
            row = i//cols+1
            col = i%cols+1
            fig.add_trace(trace, row=row, col=col)
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        autosize = True,
        margin = dict(l=0, r=0, b=0, t=30)
        )
    classes = len(figs[0].data)//2
    # 手动添加统一的图例
    for i in range(classes):
        fig.add_trace(
            go.Scatter3d(
                x=[None], y=[None], z=[None],  # 空数据
                mode='markers',
                marker=dict(color=colors[i], size=8),
                name=f'Class {i + 1}',  # 图例名称
                showlegend=True,  # 显示图例
            )
        )
    fig.update_layout(legend=dict(y=0.85))
    return fig