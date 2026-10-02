import type { ButtonHTMLAttributes, ReactNode } from "react";
import { Icon } from "./Icon";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary";
  icon?: string;
  children: ReactNode;
}

/** Pill, 56px tall. One primary per screen. */
export function Button({ variant = "primary", icon, children, className, ...rest }: Props) {
  return (
    <button className={`btn btn-${variant} ${className ?? ""}`} {...rest}>
      {icon && <Icon name={icon} color={variant === "primary" ? "white" : "ink"} size={22} />}
      {children}
    </button>
  );
}

interface LinkButtonProps {
  href: string;
  variant?: "primary" | "secondary";
  icon?: string;
  children: ReactNode;
}

export function LinkButton({ href, variant = "primary", icon, children }: LinkButtonProps) {
  return (
    <a className={`btn btn-${variant}`} href={href}>
      {icon && <Icon name={icon} color={variant === "primary" ? "white" : "ink"} size={22} />}
      {children}
    </a>
  );
}
