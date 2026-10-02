// Outline icons (2px stroke) copied from the design's icons/ set, one file per color.
export type IconColor = "ink" | "white" | "muted" | "scam" | "careful" | "normal" | "unknown";

interface Props {
  name: string;
  color?: IconColor;
  size?: number;
  alt?: string;
  className?: string;
}

export function Icon({ name, color = "ink", size = 24, alt = "", className }: Props) {
  return (
    <img
      className={className}
      src={`/icons/${name}-${color}.svg`}
      width={size}
      height={size}
      alt={alt}
      aria-hidden={alt ? undefined : true}
      draggable={false}
      style={{ display: "block", flex: "none" }}
    />
  );
}
