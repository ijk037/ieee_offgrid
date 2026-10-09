---
name: Civic Intelligence Console
colors:
  surface: '#131317'
  surface-dim: '#131317'
  surface-bright: '#39393d'
  surface-container-lowest: '#0e0e12'
  surface-container-low: '#1b1b1f'
  surface-container: '#1f1f23'
  surface-container-high: '#2a292e'
  surface-container-highest: '#353439'
  on-surface: '#e4e1e7'
  on-surface-variant: '#e2bfb0'
  inverse-surface: '#e4e1e7'
  inverse-on-surface: '#303034'
  outline: '#a98a7d'
  outline-variant: '#5a4136'
  surface-tint: '#ffb693'
  primary: '#ffb693'
  on-primary: '#561f00'
  primary-container: '#ff6b00'
  on-primary-container: '#572000'
  inverse-primary: '#a04100'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#ffb95f'
  on-tertiary: '#472a00'
  tertiary-container: '#d58800'
  on-tertiary-container: '#482b00'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#ffdbcc'
  primary-fixed-dim: '#ffb693'
  on-primary-fixed: '#351000'
  on-primary-fixed-variant: '#7a3000'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffddb8'
  tertiary-fixed-dim: '#ffb95f'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#131317'
  on-background: '#e4e1e7'
  surface-variant: '#353439'
typography:
  headline-xl:
    fontFamily: Geist
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.03em
  headline-xl-mobile:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Geist
    fontSize: 22px
    fontWeight: '500'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
  body-lg:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  metric-display:
    fontFamily: JetBrains Mono
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.04em
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-mono-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-mobile: 0.75rem
  margin: 2rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system defines an authoritative, high-velocity civic operations console built for rapid municipal response, statistical anomaly tracking, and algorithmic governance. The audience consists of city managers, emergency dispatch coordinators, municipal data scientists, and infrastructure directors who need high-signal clarity under pressure.

The aesthetic fuses **Precision High-Contrast Technical UI** with **Modern Data Surface Glass**. The interface relies on dark obsidian and deep graphite foundations, crisp architectural boundary lines, luminous focal indicators, and distinct structural density. It eliminates superfluous decoration in favor of tactical data visualization, glowing hazard tiers, and unambiguous operational states that convey total transparency and urgent civic readiness.

## Colors

The system uses an intentional dark-mode-first environment calibrated for continuous dispatch monitoring and low retinal fatigue during emergency scenarios:

- **Primary Accent (`#FF6B00`)**: Radiant Municipal Orange. Reserved exclusively for surge triggers, 3σ statistical deviations, critical civic escalations, and high-priority triage actions.
- **Secondary Accent (`#10B981`)**: Crisp Emerald Green. Communicates baseline civic equilibrium, validated SLA resolutions, healthy municipal infrastructure, and standard nominal flow.
- **Tertiary Accent (`#F59E0B`)**: Amber Gold. Encodes moderate variance, watch-tier escalations, pending field validations, and non-blocking anomalies.
- **Surfaces & Grounds**:
  - `Canvas Ground`: `#0D0D11` (pure deep obsidian)
  - `Card / Container Base`: `#18181B` (zinc deep layer)
  - `Elevated Surface`: `#27272A` (raised tactical node)
  - `Structural Border`: `#3F3F46` (active division lines)
  - `Subtle Divider`: `rgba(255, 255, 255, 0.08)`
- **Text & Foreground Metrics**:
  - `Primary Text`: `#F8FAFC`
  - `Muted Metadata`: `#A1A1AA`
  - `De-emphasized / Dim`: `#71717A`

## Typography

The typographical architecture splits cleanly between natural language processing readouts and hard telemetry metrics:

- **Primary Interface (`Geist`)**: Used across navigation, situational summaries, executive briefings, and card headers. Provides neutral, razor-sharp geometric balance that ensures zero scanning friction.
- **Data & Numerical Metrics (`JetBrains Mono`)**: Mandatory for all raw municipal telemetry, standard deviations (σ), z-scores, ward identifiers, geospatial coordinates, and timestamp logs. Monospaced tabular alignment guarantees vertical scanning integrity across rapidly shifting real-time feeds.

## Layout & Spacing

The layout utilizes a structured 12-column fluid grid on desktop displays (collapsing to 6 columns on tablet and single-column stacked modules on mobile). 

- **Grid Discipline**: Data tiles snap directly to modular spans (3-col KPIs, 4-col surge telemetry charts, 8-col geographic heatmaps).
- **Rhythm**: Compact element gaps (`space-xs` and `space-sm`) are strictly enforced within data groupings to maintain visual cohesion of related figures, while larger perimeter padding (`space-lg`) gives individual sensor modules distinct breathing room.
- **Reflow**: Tablet views preserve side-by-side metric tiles while shifting secondary event audit trails underneath primary live anomaly maps.

## Elevation & Depth

Visual hierarchy does not use soft blur shadows or heavy skeumorphic drops. Depth is maintained through layered obsidian tones, translucent glass layers, and hairline edge lighting:

- **Level 0 (Canvas Base)**: `#0D0D11` uninterrupted flat background.
- **Level 1 (Card Containers)**: `#18181B` surface bounded by a crisp 1px border (`#27272A` or `rgba(255, 255, 255, 0.08)`).
- **Level 2 (Active Surge Cards & Modals)**: `#27272A` fill accompanied by an alert rim. Critical surge containers emit a subtle targeted luminous aura: `0 0 24px -4px rgba(255, 107, 0, 0.25)`.
- **Level 3 (Tactical Overlays & Popovers)**: Translucent surface `rgba(24, 24, 27, 0.85)` treated with `backdrop-filter: blur(12px)` and bordered with `#3F3F46`.

## Shapes

The design language balances contemporary polish with technical structure:

- **Standard Containers & Cards**: Use `rounded-xl` (1.5rem / 24px) to soften dense civic data matrices and establish clear visual groupings.
- **Form Controls & Action Triggers**: Scaled at standard rounded geometry (0.5rem / 8px) to retain an agile, tool-like interaction feel.
- **Status Badges & Pill Indicators**: Rounded pill caps to distinguish qualitative classifications from rectilinear data tiles.

## Components

### Buttons
- **Critical / Primary Action**: Solid Radiant Municipal Orange (`#FF6B00`) background, `#0D0D11` bold text, sharp hover response with brightness increase (`#FF7A00`) and subtle orange bloom.
- **Secondary / Action Tool**: `#27272A` background, `#F8FAFC` text, 1px border (`#3F3F46`). Hover shifts to `#3F3F46`.
- **Ghost Action**: Transparent surface, `#A1A1AA` text, hover reveals `rgba(255, 255, 255, 0.05)` fill.

### Critical Alert Chips & Status Badges
- **Surge Badge**: Semi-transparent orange background (`rgba(255, 107, 0, 0.15)`), solid orange border (`rgba(255, 107, 0, 0.4)`), `#FF7A00` JetBrains Mono text with a pulsing 6px radial indicator dot.
- **Nominal / Resolved Badge**: Subtle green wash (`rgba(16, 185, 129, 0.12)`), `#10B981` border and text.
- **Warning Badge**: Amber wash (`rgba(245, 158, 11, 0.12)`), `#F59E0B` border and text.

### Metric Cards & Containers
- Built on `rounded-xl` geometry with deep `#18181B` backgrounds.
- Header contains an uppercase category label in `label-mono-sm` alongside an anomaly sparkline.
- Key figures are displayed using `metric-display` in JetBrains Mono, paired with a delta indicator badge showing standard deviation divergence (e.g., `+3.4σ`).

### Anomaly Incident Lists
- Segmented table rows separated by hairline borders (`rgba(255, 255, 255, 0.06)`).
- Hover states trigger a subtle horizontal wash (`rgba(255, 255, 255, 0.03)`).
- Time stamps, ward codes, and ticket ID tags are set strictly in monospace for vertical rhythm and rapid scanning.

### Input Fields & Filter Controls
- Background `#18181B` with 1px border in `#27272A`.
- Active focus state transitions border to `#FF6B00` with zero outline offset and a subtle orange glow ring (`0 0 0 1px #FF6B00`).
- Integrated clear badges and dropdown selectors follow monospaced micro-label conventions.