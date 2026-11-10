import Plotly from 'plotly.js-dist-min'
import createPlotlyComponent from 'react-plotly.js/factory'
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ComponentProps } from 'react'
const RawPlot=createPlotlyComponent(Plotly)
type Props=ComponentProps<typeof RawPlot>

/** Every chart owns its data, layout and measured dimensions, including after tab changes. */
export default function Plot(props:Props){
  const host=useRef<HTMLDivElement>(null)
  const [width,setWidth]=useState(0)
  const [error,setError]=useState('')
  const [visible,setVisible]=useState(false)
  const rawHeight=props.style?.height
  const height=typeof rawHeight==='number'?rawHeight:typeof rawHeight==='string'&&rawHeight.endsWith('px')?Number.parseFloat(rawHeight):Number(props.layout?.height??380)
  useLayoutEffect(()=>{
    const element=host.current;if(!element)return
    const observer=new ResizeObserver(entries=>{const next=Math.floor(entries[0].contentRect.width);if(next>0)setWidth(current=>current===next?current:next)})
    observer.observe(element);return()=>observer.disconnect()
  },[])
  useLayoutEffect(()=>{
    const element=host.current;if(!element)return
    const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){setVisible(true);observer.disconnect()}},{rootMargin:'120px'})
    observer.observe(element);return()=>observer.disconnect()
  },[])
  const data=useMemo(()=>structuredClone(props.data),[props.data])
  const layout=useMemo(()=>({...structuredClone(props.layout??{}),font:{family:'"Inter", "SF Pro Text", -apple-system, "Segoe UI", sans-serif',size:14,...props.layout?.font},autosize:false,width:Math.max(200,width),height}),[props.layout,width,height])
  return <div ref={host} className="responsive-plot" style={{width:'100%',minWidth:0,height,position:'relative'}}>{error?<p className="inline-error">Не удалось отобразить график: {error}</p>:visible&&width>0&&<RawPlot {...props} data={data} layout={layout} useResizeHandler={false} config={{...props.config,responsive:false,displaylogo:false}} style={{...props.style,width:'100%',height:'100%'}} onError={event=>{setError(String(event));props.onError?.(event)}}/>}</div>
}
