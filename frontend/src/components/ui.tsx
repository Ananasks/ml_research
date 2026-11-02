import type { ReactNode } from 'react'
import type { ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { motion } from 'framer-motion'
import { cn } from '../lib/utils'

const buttonStyles = cva('button inline-flex items-center justify-center gap-2 rounded-md px-3 text-xs font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:pointer-events-none disabled:opacity-50', {
  variants: {
    variant: {
      primary: 'button-primary bg-[var(--accent)] text-white',
      secondary: 'button-secondary border border-[var(--line)] bg-[var(--panel)] text-[var(--text)]',
      ghost: 'button-ghost bg-transparent text-[var(--muted)]',
      danger: 'button-danger bg-[var(--danger-soft)] text-[var(--danger)]',
    },
  },
  defaultVariants: { variant: 'primary' },
})

export function Button({ className, variant, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & VariantProps<typeof buttonStyles>) {
  return <button className={cn(buttonStyles({ variant }), className)} {...props} />
}

export function Card({ className = '', children }: { className?: string; children: ReactNode }) {
  return <motion.section initial={{ opacity: 0, y: 7 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .22 }} className={cn('card', className)}>{children}</motion.section>
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'blue' | 'green' | 'amber' | 'red' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function SectionTitle({ title, subtitle, action }: { title: string; subtitle?: string; action?: ReactNode }) {
  return <div className="section-title"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>
}
