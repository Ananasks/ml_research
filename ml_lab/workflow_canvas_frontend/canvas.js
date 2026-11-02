export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!parentElement) return;
  const viewport = parentElement.querySelector('.canvas-viewport');
  if (!viewport) return;
  viewport.style.height = `${Number(data?.height) || 600}px`;
  const world = parentElement.querySelector('.canvas-world');
  const svg = parentElement.querySelector('.canvas-edges');
  const edgeLayer = parentElement.querySelector('.canvas-edges g');
  const nodeLayer = parentElement.querySelector('.canvas-nodes');
  const graph = { nodes: structuredClone(data?.nodes || []), edges: structuredClone(data?.edges || []), selected: data?.selected || null };
  let dragging = null, panning = null, connecting = null, tempPoint = null, zoom = 1;
  let revision = 0;
  const nodeW = 174, nodeH = 174;
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs={}) => { const el=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); return el; };
  const emit = () => { revision = Date.now()*1000 + Math.floor(Math.random()*1000); setStateValue('value',{...graph,revision}); };
  const safeValues = values => Array.isArray(values) ? values.filter(Number.isFinite).slice(0,36) : [];

  function makePreview(node) {
    const preview = node.preview || {};
    const box = document.createElement('div'); box.className='node-preview';
    const s=svgEl('svg',{viewBox:'0 0 170 68',preserveAspectRatio:'none'});
    const values=safeValues(preview.values);
    const secondary=safeValues(preview.secondary);
    const label=(text,x=8,y=15)=>{const t=svgEl('text',{x,y,class:'preview-label'});t.textContent=text;s.append(t);};
    if (preview.kind==='line') {
      if (values.length) {
        const lo=Math.min(...values), hi=Math.max(...values), span=hi-lo||1;
        const pts=values.map((v,i)=>`${8+i*(154/Math.max(1,values.length-1))},${57-(v-lo)/span*43}`).join(' ');
        const area=svgEl('polygon',{points:`8,60 ${pts} 162,60`,class:'preview-fill'}); s.append(area);
        s.append(svgEl('polyline',{points:pts,class:'preview-line'}));
        if(secondary.length){const slo=Math.min(...secondary),shi=Math.max(...secondary),ss=shi-slo||1; const p2=secondary.map((v,i)=>`${8+i*(154/Math.max(1,secondary.length-1))},${57-(v-slo)/ss*43}`).join(' ');s.append(svgEl('polyline',{points:p2,class:'preview-line-alt'}));}
        label(preview.label||'Сигнал');
      } else { label('Пока нет графика'); }
    } else if(preview.kind==='model') {
      if(values.length && secondary.length) {
        const all=[...values,...secondary],lo=Math.min(...all),hi=Math.max(...all),span=hi-lo||1;
        const pointSet=series=>series.map((v,i)=>`${8+i*(154/Math.max(1,series.length-1))},${57-(v-lo)/span*43}`).join(' ');
        s.append(svgEl('polyline',{points:pointSet(values),class:'preview-line'}));
        s.append(svgEl('polyline',{points:pointSet(secondary),class:'preview-line-alt'}));
        label('Факт / прогноз');
      } else {
        // Compact model structure sketch: trees, layers or a fitted estimator profile.
        const family=preview.family||'';
        if(/нейрон|lstm|gru|rnn/i.test(family)) {
          const cols=[[22,34,46],[65,30,40,50],[112,34,46],[150,40]];
          for(let c=0;c<cols.length-1;c++) for(const y1 of cols[c]) for(const y2 of cols[c+1]) s.append(svgEl('line',{x1:25+c*43,y1,x2:65+c*43,y2,stroke:'#3978d555','stroke-width':1}));
          cols.forEach((col,c)=>col.forEach(y=>s.append(svgEl('circle',{cx:25+c*43,cy:y,r:4,fill:c===cols.length-1?'#16a36a':'#3978d5'}))));
        } else {
          for(let i=0;i<4;i++){const x=20+i*38;s.append(svgEl('line',{x1:x+8,y1:33,x2:x+28,y2:22,stroke:'#3978d577'}));s.append(svgEl('line',{x1:x+8,y1:33,x2:x+28,y2:44,stroke:'#3978d577'}));s.append(svgEl('circle',{cx:x+8,cy:33,r:5,fill:'#3978d5'}));s.append(svgEl('circle',{cx:x+29,cy:22,r:4,fill:'#16a36a'}));s.append(svgEl('circle',{cx:x+29,cy:44,r:4,fill:'#e9a23b'}));}
        }
        label(preview.family||'Модель');
      }
    } else if (preview.kind==='bars' && values.length) {
      const max=Math.max(...values.map(Math.abs))||1, gap=3, bw=(154-gap*(values.length-1))/values.length;
      values.forEach((v,i)=>s.append(svgEl('rect',{x:8+i*(bw+gap),y:59-Math.abs(v)/max*42,width:bw,height:Math.abs(v)/max*42,rx:2,class:'preview-bar'})));
      label(preview.label||'Распределение');
    } else if (preview.kind==='scatter' && values.length && secondary.length) {
      const minX=Math.min(...values), maxX=Math.max(...values), minY=Math.min(...secondary), maxY=Math.max(...secondary);
      values.forEach((v,i)=>s.append(svgEl('circle',{cx:8+(v-minX)/(maxX-minX||1)*154,cy:58-(secondary[i]-minY)/(maxY-minY||1)*44,r:2.4,class:'preview-dot'})));
      label(preview.label||'Связь признаков');
    } else if (preview.kind==='heatmap' && Array.isArray(preview.matrix)) {
      const m=preview.matrix.slice(0,6), size=10;
      m.forEach((row,r)=>row.slice(0,10).forEach((v,c)=>{const opacity=.18+Math.min(1,Math.abs(v))*0.78;s.append(svgEl('rect',{x:9+c*size,y:7+r*size,width:8,height:8,rx:1,fill:v<0?'#e9a23b':'#3978d5',opacity}));}));
      label('Корреляционная матрица',88,37);
    } else if (preview.kind==='metrics') {
      const metrics=preview.metrics||[];
      metrics.slice(0,3).forEach((m,i)=>{label(`${m[0]}  ${m[1]}`,8,20+i*17);s.append(svgEl('rect',{x:8,y:25+i*17,width:Math.max(12,Math.min(145,Number(m[2])||65)),height:3,rx:2,fill:i===0?'#3978d5':i===1?'#16a36a':'#e9a23b',opacity:.75}));});
    } else if (preview.kind==='split') {
      const chunks=preview.parts||[.6,.2,.2], colors=['#3978d5','#16a36a','#e9a23b']; let x=8;
      chunks.forEach((v,i)=>{const w=154*v;s.append(svgEl('rect',{x,y:23,width:w-2,height:20,rx:4,fill:colors[i]}));x+=w;});
      label('Train       Validation       Test');
    } else {
      const bars=values.length?values:[.35,.58,.43,.8,.52,.68,.4,.72]; const max=Math.max(...bars.map(Math.abs))||1;
      bars.slice(0,16).forEach((v,i)=>s.append(svgEl('rect',{x:9+i*9,y:59-Math.abs(v)/max*38,width:5,height:Math.abs(v)/max*38,rx:2,class:'preview-bar'})));
      label(preview.label||'Профиль признаков');
    }
    box.append(s); return box;
  }

  function pointFor(id, output) { const n=graph.nodes.find(x=>x.id===id); if(!n)return null; return {x:n.x+(output?nodeW:0),y:n.y+nodeH/2}; }
  function pathBetween(a,b) { const bend=Math.max(40,Math.abs(b.x-a.x)*.45); return `M ${a.x} ${a.y} C ${a.x+bend} ${a.y}, ${b.x-bend} ${b.y}, ${b.x} ${b.y}`; }
  function drawEdges() {
    edgeLayer.replaceChildren();
    graph.edges.forEach(edge=>{const a=pointFor(edge.source,true),b=pointFor(edge.target,false);if(!a||!b)return;
      const p=svgEl('path',{d:pathBetween(a,b),class:'edge-path','marker-end':'url(#flow-arrow)','stroke-linecap':'round'});
      p.addEventListener('click',e=>{e.stopPropagation();graph.edges=graph.edges.filter(x=>x.id!==edge.id);emit();});edgeLayer.append(p);
    });
    if(connecting&&tempPoint){const a=pointFor(connecting,true);if(a)edgeLayer.append(svgEl('path',{d:pathBetween(a,tempPoint),class:'temp-path'}));}
  }
  function render() {
    const maxX=Math.max(640,...graph.nodes.map(n=>(n.x??0)+nodeW+40));
    const maxY=Math.max(470,...graph.nodes.map(n=>(n.y??0)+nodeH+60));
    world.style.width=maxX+'px';world.style.height=maxY+'px';svg.setAttribute('width',maxX);svg.setAttribute('height',maxY);svg.style.width=maxX+'px';svg.style.height=maxY+'px';nodeLayer.style.width=maxX+'px';nodeLayer.style.height=maxY+'px';
    nodeLayer.replaceChildren();
    graph.nodes.forEach((n,i)=>{
      if(n.x==null){const row=Math.floor(i/3),column=i%3;n.x=35+(row%2?2-column:column)*200;}
      if(n.y==null)n.y=32+Math.floor(i/3)*195;
      const el=document.createElement('div');el.className='workflow-node'+(n.id===graph.selected?' selected':'');el.dataset.id=n.id;el.dataset.status=n.status||'';el.style.left=n.x+'px';el.style.top=n.y+'px';el.style.setProperty('--fill',n.fill||'#eff6ff');el.style.setProperty('--accent',n.accent||'#3978d5');
      const head=document.createElement('div');head.className='node-head';const icon=document.createElement('div');icon.className='node-icon';icon.textContent=n.icon||'•';const title=document.createElement('div');title.className='node-title';title.textContent=n.label||n.type||n.id;head.append(icon,title);
      const detail=document.createElement('div');detail.className='node-detail';detail.textContent=n.detail||'';
      const status=document.createElement('div');status.className='node-status';status.textContent=n.status||'Готов к исследованию';
      const input=document.createElement('span');input.className='port input';input.title='Вход данных';input.dataset.id=n.id;const output=document.createElement('span');output.className='port output';output.title='Выход данных';output.dataset.id=n.id;
      el.append(input,output,head,detail,makePreview(n),status);
      el.addEventListener('pointerdown',e=>{if(e.target.classList.contains('port'))return;e.preventDefault();e.stopPropagation();graph.selected=n.id;dragging={node:n,x:e.clientX,y:e.clientY,left:n.x,top:n.y,moved:false};try{parentElement.setPointerCapture(e.pointerId);}catch{}render();});
      el.addEventListener('dblclick',e=>{e.preventDefault();e.stopPropagation();graph.selected=n.id;emit();});nodeLayer.append(el);
    });
    drawEdges();
  }
  function canvasPoint(e){const r=world.getBoundingClientRect();return{x:(e.clientX-r.left)/zoom,y:(e.clientY-r.top)/zoom};}
  const onPointerDown=e=>{
    const port=e.target.closest?.('.port');
    if(port?.classList.contains('output')){connecting=port.dataset.id;tempPoint=canvasPoint(e);e.preventDefault();e.stopPropagation();drawEdges();return;}
    if(e.target===viewport||e.target===world||e.target===svg||e.target===edgeLayer){panning={x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};viewport.classList.add('panning');}
  };
  const onPointerMove=e=>{
    if(dragging){const dx=(e.clientX-dragging.x)/zoom,dy=(e.clientY-dragging.y)/zoom;if(Math.abs(dx)+Math.abs(dy)>2)dragging.moved=true;dragging.node.x=Math.max(0,dragging.left+dx);dragging.node.y=Math.max(0,dragging.top+dy);render();}
    else if(connecting){tempPoint=canvasPoint(e);drawEdges();}
    else if(panning){viewport.scrollLeft=panning.left-(e.clientX-panning.x);viewport.scrollTop=panning.top-(e.clientY-panning.y);}
  };
  const onPointerUp=e=>{
    if(dragging){const moved=dragging.moved;dragging=null;render();if(moved||graph.selected)emit();}
    if(connecting){const target=e.target.closest?.('.port.input');if(target&&target.dataset.id!==connecting&&!graph.edges.some(x=>x.source===connecting&&x.target===target.dataset.id)){graph.edges.push({id:`${connecting}-${target.dataset.id}-${Date.now()}`,source:connecting,target:target.dataset.id});}connecting=null;tempPoint=null;render();emit();}
    panning=null;viewport.classList.remove('panning');
  };
  const fitCanvas=()=>{world.style.transform='';const width=parseFloat(world.style.width)||720,height=parseFloat(world.style.height)||470;zoom=Math.min(1,(viewport.clientWidth-24)/width,(viewport.clientHeight-24)/height);zoom=Math.max(.5,zoom);world.style.transform=`scale(${zoom})`;viewport.scrollTo(0,0);};
  const onClick=e=>{const action=e.target.closest?.('[data-action]')?.dataset.action;if(!action)return;if(action==='zoom-in')zoom=Math.min(1.6,zoom+.1);if(action==='zoom-out')zoom=Math.max(.5,zoom-.1);if(action==='fit'){fitCanvas();return;}world.style.transform=`scale(${zoom})`;};
  parentElement.addEventListener('pointerdown',onPointerDown,true);parentElement.addEventListener('pointermove',onPointerMove);parentElement.addEventListener('pointerup',onPointerUp);parentElement.addEventListener('click',onClick);render();fitCanvas();
  return ()=>{parentElement.removeEventListener('pointerdown',onPointerDown,true);parentElement.removeEventListener('pointermove',onPointerMove);parentElement.removeEventListener('pointerup',onPointerUp);parentElement.removeEventListener('click',onClick);};
}
