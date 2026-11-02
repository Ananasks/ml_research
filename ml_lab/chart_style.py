"""Shared per-chart style editor. Styling copies figures and never changes data."""
import copy
import re
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PALETTE=['#2563eb','#f97316','#16a34a','#9333ea','#0891b2','#e11d48','#ca8a04','#4f46e5']

THEMES={'Исходная':None,'Светлая':'plotly_white','Тёмная':'plotly_dark','Минимализм':'simple_white','Научная':'none'}
DASHES={'Исходный':None,'Сплошная':'solid','Штриховая':'dash','Точечная':'dot','Штрихпунктирная':'dashdot'}
SCALES=['Исходная','Viridis','Cividis','Plasma','Blues','RdBu_r','Turbo']


def axis_names(figure):
    return sorted({'xaxis','yaxis'} | {key for key in figure.layout.to_plotly_json() if re.fullmatch(r'[xy]axis\d*',key)})


def apply_style(figure, settings):
    styled=go.Figure(figure)
    styled.update_layout(template='plotly_white',colorway=PALETTE,
        font=dict(family='Arial, sans-serif',size=14,color='#24344b'),
        paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',hoverlabel=dict(font_size=13))
    styled.update_xaxes(automargin=True)
    styled.update_yaxes(automargin=True,gridcolor='#e2e8f0')
    if settings.get('theme') in THEMES and THEMES[settings['theme']]:
        styled.update_layout(template=THEMES[settings['theme']])
        if settings['theme']=='Тёмная':
            styled.update_layout(paper_bgcolor='#111827',plot_bgcolor='#111827',font_color='#e5e7eb')
    if 'title' in settings: styled.update_layout(title_text=settings['title'])
    if 'legend' in settings: styled.update_layout(showlegend=settings['legend'])
    if 'font_size' in settings: styled.update_layout(font_size=settings['font_size'])
    for name,label in settings.get('axes',{}).items():
        styled.layout[name].title.text=label
    grid=settings.get('grid','Исходная')
    if grid!='Исходная':
        for name in axis_names(styled):
            axis=styled.layout[name]
            axis.showgrid=grid=='Обе оси' or grid==('Только X' if name.startswith('x') else 'Только Y')
            axis.gridcolor=settings.get('grid_color','#dce3eb')
            axis.gridwidth=settings.get('grid_width',1)
    scale=settings.get('scale','Исходная')
    if scale!='Исходная':
        for name in styled.layout.to_plotly_json():
            if re.fullmatch(r'coloraxis\d*',name): styled.layout[name].colorscale=scale
    for i,trace in enumerate(styled.data):
        trace_type=trace.type
        if trace_type in ('heatmap','contour'):
            if scale!='Исходная': trace.colorscale=scale
            continue
        color=settings.get('colors',{}).get(str(i)) if settings.get('custom_colors') else None
        if trace_type in ('scatter','scattergl'):
            if color: trace.line.color=color; trace.marker.color=color
            if settings.get('line_width') is not None: trace.line.width=settings['line_width']
            dash=DASHES.get(settings.get('dash'))
            if dash: trace.line.dash=dash
            mode=settings.get('mode','Исходный')
            if mode!='Исходный': trace.mode={'Линии':'lines','Точки':'markers','Линии и точки':'lines+markers'}[mode]
            if settings.get('marker_size') is not None: trace.marker.size=settings['marker_size']
        elif trace_type in ('bar','histogram','box','violin') and color:
            trace.marker.color=color
            if trace_type in ('box','violin'): trace.line.color=color
        elif trace_type=='waterfall' and settings.get('custom_colors'):
            trace.increasing.marker.color=settings.get('positive_color','#16a34a')
            trace.decreasing.marker.color=settings.get('negative_color','#dc2626')
            trace.totals.marker.color=settings.get('total_color','#2563eb')
    return styled


def _reset_style(key):
    st.session_state.setdefault('chart_styles',{}).pop(key,None)
    prefix='chart_style_widget_'+key+'__'
    for name in list(st.session_state):
        if name.startswith(prefix): del st.session_state[name]


def _hex_color(value, fallback):
    return value if isinstance(value,str) and re.fullmatch(r'#[0-9a-fA-F]{6}',value) else fallback


def show_chart(figure, *, key):
    """A form avoids rerunning expensive charts for each individual style edit."""
    styles=st.session_state.setdefault('chart_styles',{})
    saved=styles.get(key,{})
    prefix='chart_style_widget_'+key+'__'
    def widget_key(name, default):
        name=prefix+name
        st.session_state.setdefault(name,default)
        return name
    title=figure.layout.title.text or 'График'
    with st.expander('Оформление · '+title):
        with st.form('chart_style_form_'+key):
            settings={}
            settings['title']=st.text_input('Заголовок графика',key=widget_key('title',saved.get('title',title)))
            settings['axes']={}
            axes=axis_names(figure)
            columns=st.columns(2)
            for i,name in enumerate(axes):
                default=saved.get('axes',{}).get(name,figure.layout[name].title.text or '')
                label=('Подпись оси X' if name.startswith('x') else 'Подпись оси Y')
                if len(axes)>2: label+=' · '+name
                settings['axes'][name]=columns[i%2].text_input(label,key=widget_key(name,default))
            a,b,c=st.columns(3)
            settings['grid']=a.selectbox('Сетка',['Исходная','Нет','Обе оси','Только X','Только Y'],key=widget_key('grid',saved.get('grid','Исходная')))
            settings['theme']=b.selectbox('Стиль оформления',list(THEMES),key=widget_key('theme',saved.get('theme','Исходная')))
            settings['font_size']=c.number_input('Размер шрифта',8,32,key=widget_key('font_size',saved.get('font_size',14)))
            settings['grid_color']=a.color_picker('Цвет сетки',key=widget_key('grid_color',saved.get('grid_color','#dce3eb')))
            settings['grid_width']=b.number_input('Толщина сетки',.5,4.,step=.5,key=widget_key('grid_width',saved.get('grid_width',1.)))
            settings['legend']=c.checkbox('Показывать легенду',key=widget_key('legend',saved.get('legend',figure.layout.showlegend is not False)))
            has_lines=any(trace.type in ('scatter','scattergl') for trace in figure.data)
            if has_lines:
                a,b,c=st.columns(3)
                settings['mode']=a.selectbox('Линии и маркеры',['Исходный','Линии','Точки','Линии и точки'],key=widget_key('mode',saved.get('mode','Исходный')))
                settings['dash']=b.selectbox('Стиль линии',list(DASHES),key=widget_key('dash',saved.get('dash','Исходный')))
                settings['line_width']=c.number_input('Толщина линии',.5,8.,step=.5,key=widget_key('line_width',saved.get('line_width',2.)))
                settings['marker_size']=c.number_input('Размер маркера',2,20,key=widget_key('marker_size',saved.get('marker_size',6)))
            if any(trace.type in ('heatmap','contour') for trace in figure.data):
                settings['scale']=st.selectbox('Шкала цвета',SCALES,key=widget_key('scale',saved.get('scale','Исходная')))
            else:
                settings['custom_colors']=st.checkbox('Использовать выбранные цвета',key=widget_key('custom_colors',saved.get('custom_colors',False)))
                settings['colors']={}
                columns=st.columns(3)
                for i,trace in enumerate(figure.data):
                    if trace.type not in ('scatter','scattergl','bar','histogram','box','violin'): continue
                    original=getattr(getattr(trace,'line',None),'color',None)
                    if original is None: original=getattr(getattr(trace,'marker',None),'color',None)
                    default=_hex_color(original,px.colors.qualitative.Plotly[i%10])
                    settings['colors'][str(i)]=columns[i%3].color_picker('Цвет · '+(trace.name or str(i+1)),key=widget_key('color_'+str(i),saved.get('colors',{}).get(str(i),default)))
                if any(trace.type=='waterfall' for trace in figure.data):
                    for column,name,label,default in zip(columns,['positive_color','negative_color','total_color'],['Положительный вклад','Отрицательный вклад','Итог'],['#16a34a','#dc2626','#2563eb']):
                        settings[name]=column.color_picker(label,key=widget_key(name,saved.get(name,default)))
            a,b=st.columns(2)
            submitted=a.form_submit_button('Применить оформление')
            b.form_submit_button('Сбросить оформление',on_click=_reset_style,args=(key,))
        if submitted:
            styles[key]=copy.deepcopy(settings)
            saved=styles[key]
        st.caption('Настройки отдельные для каждого графика. Измените параметры и нажмите «Применить оформление».')
    styled=apply_style(figure,saved)
    st.plotly_chart(styled,width='stretch',key=key,theme=None)
    return styled
