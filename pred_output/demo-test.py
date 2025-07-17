import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, no_update
import numpy as np
import os
from datetime import datetime
import plotly.io as pio
from sklearn.metrics import r2_score
import warnings
from dash.exceptions import PreventUpdate
import base64
from io import BytesIO

# 禁用警告
warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"

# 配置参数
SITE_NAME = "B99"
DATA_PATH = f"./{SITE_NAME}/{SITE_NAME}_pred1by1.csv"
OUTPUT_DIR = "./html"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "background": "#f5f5f5",
    "text": "#333333",
    "success": "#28a745",
    "warning": "#ffc107"
}
POD_TS_THRESHOLD = 0.1

# 加载数据并处理NaN值
try:
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # 处理NaN值 - 删除包含NaN的行
    df = df.dropna(subset=['true', 'pred'])

    # 检查数据是否为空
    if df.empty:
        raise ValueError("数据为空，请检查数据文件")

except Exception as e:
    raise RuntimeError(f"数据加载失败: {str(e)}")

# 初始化Dash应用
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = f"{SITE_NAME}预测分析"
server = app.server


# 增强的calculate_metrics函数
def calculate_metrics(df_subset):
    # 默认返回值
    default_metrics = {
        "rmse": np.nan,
        "r2": np.nan,
        "pod": np.nan,
        "ts": np.nan,
        "far": np.nan,
        "date_range": "无数据",
        "count": 0
    }

    if df_subset.empty:
        return default_metrics

    # 确保没有NaN值
    df_subset = df_subset.dropna(subset=['true', 'pred'])
    if df_subset.empty:
        return default_metrics

    try:
        # 计算日期范围字符串
        date_min = df_subset['date'].min()
        date_max = df_subset['date'].max()
        date_range_str = f"{date_min.strftime('%Y-%m-%d')} 至 {date_max.strftime('%Y-%m-%d')}"

        # 计算RMSE
        rmse = np.sqrt(np.mean((df_subset['true'] - df_subset['pred']) ** 2))

        # 计算R² - 需要至少2个样本
        r2 = np.nan
        if len(df_subset) > 1:
            try:
                r2 = r2_score(df_subset['true'], df_subset['pred'])
            except:
                r2 = np.nan

        # 计算二分类指标
        true_bin = (df_subset['true'] > POD_TS_THRESHOLD).astype(int)
        pred_bin = (df_subset['pred'] > POD_TS_THRESHOLD).astype(int)

        TP = ((true_bin == 1) & (pred_bin == 1)).sum()
        FN = ((true_bin == 1) & (pred_bin == 0)).sum()
        FP = ((true_bin == 0) & (pred_bin == 1)).sum()

        pod = TP / (TP + FN) if (TP + FN) > 0 else np.nan
        ts = TP / (TP + FN + FP) if (TP + FN + FP) > 0 else np.nan
        far = FP / (TP + FP) if (TP + FP) > 0 else np.nan

        return {
            "rmse": rmse,
            "r2": r2,
            "pod": pod,
            "ts": ts,
            "far": far,
            "date_range": date_range_str,
            "count": len(df_subset)
        }
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return default_metrics


# 创建图表函数
def create_simple_figure(start_date, end_date):
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df_subset = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    except:
        df_subset = pd.DataFrame()  # 如果日期解析失败，返回空DataFrame

    fig = go.Figure()

    if not df_subset.empty:
        fig.add_trace(go.Bar(
            x=df_subset['date'],
            y=df_subset['true'],
            name='真实值',
            marker_color=COLORS['primary'],
            opacity=0.7
        ))
        fig.add_trace(go.Bar(
            x=df_subset['date'],
            y=df_subset['pred'],
            name='预测值',
            marker_color=COLORS['secondary'],
            opacity=0.7
        ))
    else:
        # 添加空数据提示
        fig.add_annotation(
            text="没有可用数据",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )

    # 格式化日期显示
    start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
    end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)

    fig.update_layout(
        title=f'{SITE_NAME} - 真实值与预测值对比 ({start_str} 至 {end_str})',
        xaxis_title='日期',
        yaxis_title='数值',
        barmode='group',
        hovermode='x unified',
        margin=dict(l=40, r=20, t=80, b=40),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(rangeslider=dict(visible=True)),
        height=700
    )

    return fig


# 应用布局
app.layout = html.Div([
    html.H1(f"{SITE_NAME}预测分析", style={'textAlign': 'center'}),

    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=df['date'].min(),
        max_date_allowed=df['date'].max(),
        start_date=df['date'].min(),
        end_date=df['date'].max(),
        display_format='YYYY-MM-DD',
        style={'margin': '20px auto', 'display': 'block'}
    ),

    html.Div(id='metrics-container', style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(4, 1fr)',
        'gap': '15px',
        'margin': '20px',
        'padding': '15px',
        'border': '1px solid #ddd',
        'borderRadius': '5px'
    }),

    dcc.Graph(
        id='main-chart',
        config={'displayModeBar': True},
        style={'height': '80vh'}
    ),

    html.Div([
        html.Button(
            '保存当前视图',
            id='save-button',
            style={
                'backgroundColor': COLORS['primary'],
                'color': 'white',
                'padding': '10px 20px',
                'border': 'none',
                'borderRadius': '4px',
                'cursor': 'pointer',
                'margin': '20px'
            }
        ),
        html.Div(id='save-status', style={'margin': '20px'})
    ], style={'textAlign': 'center'})
])


# 增强的回调函数
@app.callback(
    [Output('main-chart', 'figure'),
     Output('metrics-container', 'children')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_chart(start_date, end_date):
    if not start_date or not end_date:
        raise PreventUpdate

    try:
        # 尝试转换日期格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 获取数据子集
        df_subset = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        metrics = calculate_metrics(df_subset)

        # 创建图表
        fig = create_simple_figure(start_date, end_date)

        # 创建指标显示 - 确保所有键都存在
        metrics_display = [
            html.Div([
                html.Div("日期范围", style={'fontWeight': 'bold'}),
                html.Div(metrics.get('date_range', 'N/A'))
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']}),

            html.Div([
                html.Div("RMSE", style={'fontWeight': 'bold'}),
                html.Div(f"{metrics.get('rmse', np.nan):.3f}" if not np.isnan(metrics.get('rmse', np.nan)) else "N/A")
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']}),

            html.Div([
                html.Div("R²", style={'fontWeight': 'bold'}),
                html.Div(f"{metrics.get('r2', np.nan):.3f}" if not np.isnan(metrics.get('r2', np.nan)) else "N/A")
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']}),

            html.Div([
                html.Div(f"POD (>{POD_TS_THRESHOLD})", style={'fontWeight': 'bold'}),
                html.Div(f"{metrics.get('pod', np.nan):.3f}" if not np.isnan(metrics.get('pod', np.nan)) else "N/A")
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']}),

            html.Div([
                html.Div(f"TS (>{POD_TS_THRESHOLD})", style={'fontWeight': 'bold'}),
                html.Div(f"{metrics.get('ts', np.nan):.3f}" if not np.isnan(metrics.get('ts', np.nan)) else "N/A")
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']}),

            html.Div([
                html.Div(f"FAR (>{POD_TS_THRESHOLD})", style={'fontWeight': 'bold'}),
                html.Div(f"{metrics.get('far', np.nan):.3f}" if not np.isnan(metrics.get('far', np.nan)) else "N/A")
            ], style={'padding': '10px', 'backgroundColor': COLORS['background']})
        ]

        return fig, metrics_display

    except Exception as e:
        print(f"更新图表时出错: {str(e)}")
        # 返回空图表和错误信息
        error_fig = go.Figure()
        error_fig.add_annotation(
            text="数据加载出错",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        error_fig.update_layout(height=700)

        error_message = html.Div(
            "数据加载出错，请检查日期选择",
            style={'color': COLORS['warning'], 'padding': '20px', 'textAlign': 'center'}
        )

        return error_fig, error_message


# 保存当前视图回调
@app.callback(
    Output('save-status', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('date-picker', 'start_date'),
     State('date-picker', 'end_date')],
    prevent_initial_call=True
)
def save_current_view(n_clicks, start_date, end_date):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    try:
        # 创建图表
        fig = create_simple_figure(start_date, end_date)
        df_subset = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        metrics = calculate_metrics(df_subset)

        # 生成图片
        img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
        img_data = base64.b64encode(img_bytes).decode('utf-8')

        # 生成HTML报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{SITE_NAME}_{timestamp}.html"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # 格式化指标显示
        def format_metric(value):
            if np.isnan(value):
                return 'N/A'
            return f"{value:.3f}"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{SITE_NAME}分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
                .metric {{ padding: 10px; background: #f5f5f5; border-radius: 5px; }}
                .chart-container {{ margin: 30px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{SITE_NAME}预测分析报告</h1>
                    <h3>{metrics.get('date_range', '无日期范围信息')}</h3>
                </div>

                <div class="metrics">
                    <div class="metric"><strong>RMSE</strong>: {format_metric(metrics.get('rmse', np.nan))}</div>
                    <div class="metric"><strong>R²</strong>: {format_metric(metrics.get('r2', np.nan))}</div>
                    <div class="metric"><strong>POD</strong>: {format_metric(metrics.get('pod', np.nan))}</div>
                    <div class="metric"><strong>TS</strong>: {format_metric(metrics.get('ts', np.nan))}</div>
                    <div class="metric"><strong>FAR</strong>: {format_metric(metrics.get('far', np.nan))}</div>
                </div>

                <div class="chart-container">
                    <img src="data:image/png;base64,{img_data}" alt="分析图表">
                </div>

                <div style="margin-top: 30px; font-size: 0.8em; color: #666; text-align: center;">
                    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """

        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html.Div(
            "报告已成功保存！",
            style={'color': COLORS['success'], 'fontWeight': 'bold'}
        )

    except Exception as e:
        return html.Div(
            f"保存失败: {str(e)}",
            style={'color': COLORS['warning'], 'fontWeight': 'bold'}
        )


if __name__ == '__main__':
    # 打印数据信息用于调试
    print(f"数据加载成功，时间范围: {df['date'].min()} 到 {df['date'].max()}")
    # print(f"数据点数: {len(df)}")

    app.run(debug=True)