import {createElement as el,type ReactNode} from 'react'
// Native MathML: selectable, accessible mathematics without HTML injection.
const node=(tag:string,...children:ReactNode[])=>el(tag,null,...children)
const row=(...children:ReactNode[])=>node('mrow',...children)
const v=(name:string)=>node('mi',name),num=(value:string|number)=>node('mn',String(value)),op=(value:string)=>node('mo',value)
const text=(value:string)=>node('mtext',value)
const sub=(base:ReactNode,index:ReactNode)=>node('msub',base,index)
const sup=(base:ReactNode,power:ReactNode)=>node('msup',base,power)
const frac=(top:ReactNode,bottom:ReactNode)=>node('mfrac',top,bottom)
const root=(value:ReactNode)=>node('msqrt',value)
const par=(value:ReactNode)=>row(op('('),value,op(')'))
const abs=(value:ReactNode)=>row(op('|'),value,op('|'))
const eq=(a:ReactNode,b:ReactNode)=>row(a,op('='),b)
const fn=(name:string,arg:ReactNode)=>row(v(name),par(arg))
const exp=(power:ReactNode)=>sup(v('e'),power)
const sum=(index='ℓ')=>node('munderover',op('∑'),row(v(index),op('='),num(0)),row(v('L'),op('−'),num(1)))
const fc=sub(v('f'),v('c')),fd=sub(v('f'),v('D')),rF=sub(v('r'),v('F'))
const S=fn('S',v('t')),stec=fn('STEC',v('t')),phase=sub(v('φ'),text('adv'))
const path=frac(row(num('40.3'),S),sup(fc,num(2)))
const phi=frac(row(num(2),v('π'),num('40.3'),S),row(v('c'),fc))
const L0=sub(v('L'),num(0)),l0=sub(v('l'),num(0)),dx=row(v('Δ'),v('x'))
const sigma=sub(v('σ'),v('φ')),fs=sub(v('F'),v('s'))
const Hion=fn('H',row(v('f'),op(','),v('t')))
const scatter=row(sum(),sub(v('b'),v('ℓ')),par(v('t')),exp(row(op('−'),v('j'),num(2),v('π'),v('ν'),sub(v('τ'),v('ℓ')))))
export const radioExpressions={
 mapping:eq(fn('M',v('E')),sup(par(row(num(1),op('−'),sup(frac(row(sub(v('R'),v('E')),fn('cos',v('E'))),row(sub(v('R'),v('E')),op('+'),sub(v('h'),v('I')))),num(2)))),frac(row(op('−'),num(1)),num(2)))),
 stecCompact:eq(v('STEC'),row(fn('M',v('E')),v('VTEC'))),
 outerScale:eq(L0,row(num(4),rF)),
 innerScale:eq(l0,row(num('0.4'),rF)),
 index:eq(v('p'),frac(num(11),num(3))),
 gridPower:eq(v('N'),sup(num(2),v('m'))),
 gridDomain:row(v('N'),dx,op('≥'),num(4),L0),
 gridTrack:row(v('v'),v('T'),op('≤'),num('0.8'),v('N'),dx),
 stec:eq(stec,row(fn('M',v('E')),fn('VTEC',v('t')))),
 electrons:eq(S,row(stec,op('·'),sup(num(10),num(16)))),
 path:eq(sub(v('ΔL'),v('g')),path),
 delay:eq(fn('τ',row(v('f'),op(','),v('t'))),frac(row(num('40.3'),S),row(v('c'),sup(v('f'),num(2))))),
 phase:eq(phase,phi),
 doppler:eq(sub(v('f'),text('ion')),row(frac(num('40.3'),row(v('c'),fc)),frac(row(text('d'),S),row(text('d'),v('t'))))),
 phaseRms:eq(sigma,row(frac(row(num(2),v('π'),num('40.3')),row(v('c'),fc)),sub(v('σ'),text('STEC')))),
 fresnel:eq(rF,root(frac(row(v('λ'),v('z')),row(num(2),v('π'))))),
 wavelength:eq(v('λ'),frac(v('c'),fc)),
 scales:row(eq(L0,row(num(4),rF)),op(','),eq(l0,row(num('0.4'),rF)),op(','),eq(v('p'),frac(num(11),num(3)))),
 spacing:eq(dx,fn('min',row(frac(l0,num(2)),op(','),frac(rF,num(3)),op(','),num(100),text(' м')))),
 grid:row(eq(v('N'),sup(num(2),v('m'))),op(','),v('N'),dx,op('≥'),num(4),L0,op(','),v('v'),v('T'),op('≤'),num('0.8'),v('N'),dx),
 sampleClarke:eq(fs,fn('min',row(num(160),op(','),fn('max',row(num(40),op(','),op('⌈'),num(20),fd,op('⌉')))))),
 sampleScreen:eq(fs,fn('min',row(num(160),op(','),fn('max',row(num(40),op(','),op('⌈'),frac(row(num(4),v('v')),dx),op('⌉')))))),
 sampleFixed:eq(fs,num(40)),
 sampleCount:eq(v('M'),row(op('⌊'),v('T'),fs,op('⌋'))),
 motion:eq(fd,frac(v('v'),v('λ'))),
 clarke:eq(row(sub(v('g'),v('ℓ')),par(v('t'))),row(frac(num(1),root(v('N'))),node('munderover',op('∑'),row(v('k'),op('='),num(1)),v('N')),exp(row(v('j'),par(row(num(2),v('π'),fd,fn('cos',sub(v('α'),row(v('ℓ'),v('k')))),v('t'),op('+'),sub(v('ψ'),row(v('ℓ'),v('k'))))))))),
 arrival:eq(sub(v('f'),row(v('D'),op(','),v('k'))),row(fd,fn('cos',sub(v('α'),v('k'))))),
 acf:eq(row(sub(v('R'),v('g')),par(v('Δt'))),row(sub(v('J'),num(0)),par(row(num(2),v('π'),fd,v('Δt'))))),
 psdEstimate:eq(node('mover',sub(v('S'),v('h')),op('^')),row(text('Welch'),par(fn('h',v('t'))))),
 acfEstimate:eq(node('mover',v('R'),op('^')),frac(row(op('∑'),par(row(sub(v('h'),row(v('n'),op('+'),v('k'))),op('−'),v('μ'))),sup(par(row(sub(v('h'),v('n')),op('−'),v('μ'))),op('*'))),row(par(row(v('M'),op('−'),v('k'))),sup(v('σ'),num(2))))),
 jakes:row(eq(row(sub(v('S'),v('g')),par(v('f'))),frac(num(1),row(v('π'),fd,root(row(num(1),op('−'),sup(par(frac(v('f'),fd)),num(2))))))),op(','),abs(v('f')),op('<'),fd),
 multipath:eq(fn('H',row(v('ν'),op(','),v('t'))),scatter),
 spectrum:row(sub(v('Φ'),v('φ')),par(v('q')),op('∝'),sup(par(row(sup(v('q'),num(2)),op('+'),sup(L0,row(op('−'),num(2))))),frac(row(op('−'),v('p')),num(2))),exp(row(op('−'),sup(par(row(v('q'),l0)),num(2))))),
 screen:eq(row(sub(v('U'),num(0)),par(row(v('x'),op(','),v('y')))),row(exp(row(v('j'),fn('φ',row(v('x'),op(','),v('y'))))))),
 fresnelPropagation:eq(sub(v('U'),v('z')),row(sup(sub(v('ℱ'),num(2)),row(op('−'),num(1))),par(row(sub(v('ℱ'),num(2)),par(exp(row(v('j'),v('φ')))),exp(row(v('j'),v('π'),v('λ'),v('z'),par(row(sup(sub(v('q'),v('x')),num(2)),op('+'),sup(sub(v('q'),v('y')),num(2)))))))))),
 track:eq(fn('h',v('t')),row(sub(v('U'),v('z')),par(row(sub(v('x'),num(0)),op('+'),v('v'),v('t'),op(','),sub(v('y'),num(0)))))),
 structure:eq(row(sub(v('D'),v('φ')),par(v('ρ'))),row(text('E'),par(sup(par(row(fn('φ',row(v('r'),op('+'),v('ρ'))),op('−'),fn('φ',v('r')))),num(2))))),
 hybrid:eq(Hion,row(sub(v('H'),text('ion')),par(row(v('f'),op(','),v('t'))),op('·'),sub(v('H'),text('scat')),par(row(v('f'),op(','),v('t'))))),
 ionosphere:eq(row(sub(v('H'),text('ion')),par(row(v('f'),op(','),v('t')))),row(exp(row(v('j'),frac(row(num(2),v('π'),num('40.3'),S),row(v('c'),v('f'))))))),
 localTec:eq(S,row(sup(num(10),num(16)),par(row(sub(v('STEC'),num(0)),op('+'),frac(row(text('d'),v('STEC')),row(text('d'),v('t'))),v('t'))))),
 amplitude:eq(abs(v('h')),root(row(sup(fn('Re',v('h')),num(2)),op('+'),sup(fn('Im',v('h')),num(2))))),
 channelPhase:eq(v('φ'),fn('unwrap',fn('atan2',row(fn('Im',v('h')),op(','),fn('Re',v('h')))))),
 components:eq(v('h'),row(fn('Re',v('h')),op('+'),v('j'),fn('Im',v('h')))),
 gain:eq(sub(v('A'),text('dB')),row(num(20),sub(v('log'),num(10)),abs(fn('H',row(fc,op('+'),v('ν'),op(','),sub(v('t'),num(0))))))),
 responsePhase:eq(v('Φ'),fn('unwrap',row(op('arg'),fn('H',row(fc,op('+'),v('ν'),op(','),sub(v('t'),num(0))))))),
 cir:eq(row(sub(v('h'),v('B')),par(v('τ'))),row(sup(v('ℱ'),row(op('−'),num(1))),par(fn('H',row(fc,op('+'),v('ν'),op(','),sub(v('t'),num(0))))))),
 correlation:eq(row(sub(v('R'),v('H')),par(v('Δf'))),row(frac(row(sum(),sub(v('P'),v('ℓ')),exp(row(op('−'),v('j'),num(2),v('π'),v('Δf'),sub(v('τ'),v('ℓ'))))),row(sum(),sub(v('P'),v('ℓ')))))),
 profile:row(eq(sub(v('τ'),v('ℓ')),frac(row(v('ℓ'),sub(v('τ'),text('max'))),row(v('L'),op('−'),num(1)))),op(','),sub(v('P'),v('ℓ')),op('∝'),exp(frac(row(op('−'),num(3),v('ℓ')),row(v('L'),op('−'),num(1))))),
 s4:eq(sub(v('S'),num(4)),frac(fn('std',sup(abs(v('h')),num(2))),fn('mean',sup(abs(v('h')),num(2))))),
 power:eq(v('I'),sup(abs(v('h')),num(2))),
}
export type RadioExpression=keyof typeof radioExpressions
export function RadioMath({kind,inline=false}:{kind:RadioExpression;inline?:boolean}){return <span className={`radio-math ${inline?'radio-math-inline':''}`}>{el('math',{display:inline?'inline':'block','aria-label':kind},radioExpressions[kind])}</span>}
export function RadioMathGroup({kinds}:{kinds:RadioExpression[]}){return <div className="radio-math-group">{kinds.map(kind=><RadioMath key={kind} kind={kind}/>)}</div>}
export const parameterMath:Partial<Record<string,RadioExpression>>={mapping:'mapping',stec_tecu:'stec',path_m:'path',delay_ns:'delay',phase_rad:'phase',iono_doppler_hz:'doppler',fresnel_m:'fresnel',outer_scale_m:'outerScale',inner_scale_m:'innerScale',spacing_m:'spacing',screen_size:'grid',samples:'sampleCount'}
const plotMath:Record<string,RadioExpression>={'Огибающая канала':'amplitude','Фаза канала':'channelPhase','Комплексные компоненты':'components','Автокорреляция':'acfEstimate','Доплеровский спектр':'psdEstimate','Частотная корреляция':'correlation','Амплитудно-частотная характеристика':'gain','Фазочастотная характеристика':'responsePhase','Полосоограниченная импульсная характеристика':'cir','Профиль мощности заданных лучей':'profile','Структурная функция фазы':'structure','Рассчитанный фазовый экран':'spectrum','Интенсивность после распространения':'fresnelPropagation','Групповая задержка по ПЭС':'delay','Фазовое опережение по ПЭС':'phase'}
export function RadioPlotFormula({title,formula}:{title:string;formula:string}){const kind=plotMath[title];return kind?<RadioMath kind={kind}/>:<p className="radio-math-context">{formula}</p>}

const symbols:Record<string,ReactNode>={STEC:v('STEC'),τg:sub(v('τ'),v('g')),φadv:sub(v('φ'),text('adv')),σφ:sigma,rF,Fs:fs,'L₀, l₀':row(L0,op(','),l0),'L₀, l₀, p':row(L0,op(','),l0,op(','),v('p')),'Δx, N':row(dx,op(','),v('N')),'Δx':dx,N:v('N')}
export function RadioSymbol({name}:{name:string}){return symbols[name]?el('math',{display:'inline','aria-label':name},symbols[name]):<>{name}</>}
