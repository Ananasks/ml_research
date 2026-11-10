import {createElement as el,type ReactNode} from 'react'
const m=(tag:string,...children:ReactNode[])=>el(tag,null,...children)
const v=(s:string)=>m('mi',s), n=(s:string)=>m('mn',s), o=(s:string)=>m('mo',s)
const sub=(a:string,b:string)=>m('msub',v(a),v(b))
const frac=(a:ReactNode,b:ReactNode)=>m('mfrac',a,b)
const row=(...a:ReactNode[])=>m('mrow',...a)
const pow=(a:ReactNode,b:string)=>m('msup',a,n(b))
const S=row(v('S'),o('('),v('t'),o(')'))
export function ChannelEquation({kind,wide=false}:{kind:'delay'|'phase'|'doppler'|'clarke'|'transfer'|'spread';wide?:boolean}){
 const f=wide?v('f'):sub('f','c'), phase=frac(row(n('2'),v('π'),o('·'),n('40.3'),S),row(v('c'),f))
 let expression:ReactNode
 if(kind==='delay')expression=row(sub('τ','g'),o('('),wide?row(v('f'),o(',')):null,v('t'),o(')'),o('='),frac(row(n('40.3'),S),row(v('c'),pow(f,'2'))))
 else if(kind==='phase')expression=row(sub('φ','adv'),o('('),wide?row(v('f'),o(',')):null,v('t'),o(')'),o('='),phase)
 else if(kind==='doppler')expression=row(sub('f','iono'),o('('),v('t'),o(')'),o('='),frac(n('40.3'),row(v('c'),sub('f','c'))),frac(row(o('d'),S),row(o('d'),v('t'))))
 else if(kind==='clarke')expression=null
 else if(kind==='spread')expression=row(sub('Δτ','B'),o('('),v('t'),o(')'),o('='),o('|'),sub('τ','g'),o('('),sub('f','c'),o('−'),frac(v('B'),n('2')),o(','),v('t'),o(')'),o('−'),sub('τ','g'),o('('),sub('f','c'),o('+'),frac(v('B'),n('2')),o(','),v('t'),o(')'),o('|'))
 else expression=wide?row(v('H'),o('('),v('f'),o(','),v('t'),o(')'),o('='),sub('g','C'),o('('),v('t'),o(')'),m('msup',v('e'),row(v('j'),phase))):row(v('h'),o('('),v('t'),o(')'),o('='),sub('g','C'),o('('),v('t'),o(')'),m('msup',v('e'),row(v('j'),sub('φ','adv'),o('('),v('t'),o(')'))))
 if(kind==='clarke')expression=row(sub('g','C'),o('('),v('t'),o(')'),o('='),frac(n('1'),m('msqrt',v('N'))),m('munderover',o('∑'),row(v('k'),o('='),n('1')),v('N')),m('msup',v('e'),row(v('j'),o('['),n('2'),v('π'),sub('f','D'),o('cos'),sub('α','k'),v('t'),o('+'),sub('ψ','k'),o(']'))))
 const title=({delay:"Групповая задержка",phase:"Фазовое опережение",doppler:"Ионосферный частотный сдвиг",clarke:"Сумма волн Кларка",transfer:"Комплексный коэффициент канала",spread:"Разность задержек на краях полосы"})[kind]
 return <div className="channel-equation"><span className="channel-equation-title">{title}</span>{el('math',{'display':'block','aria-label':({delay:'Групповая задержка',phase:'Фазовое опережение',doppler:'Ионосферный частотный сдвиг',clarke:'Сумма волн Кларка',transfer:'Передаточная функция канала',spread:'Разность задержек на краях полосы'})[kind]},expression)}</div>
}
export default function ChannelEquations({wide=false}:{wide?:boolean}){return <div className="channel-equations"><ChannelEquation kind="delay" wide={wide}/><ChannelEquation kind="phase" wide={wide}/><ChannelEquation kind="doppler"/>{wide&&<ChannelEquation kind="spread"/>}</div>}
