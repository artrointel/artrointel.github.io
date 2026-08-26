export const locales = ['en', 'ko'] as const;

export type Locale = (typeof locales)[number];

export const defaultLocale: Locale = 'en';

export const isLocale = (value: unknown): value is Locale =>
  typeof value === 'string' && locales.includes(value as Locale);

export const getLocalizedPath = (pathname: string, locale: Locale): string => {
  const normalized = pathname.startsWith('/') ? pathname : `/${pathname}`;
  const withoutLocale = normalized.replace(/^\/(en|ko)(?=\/|$)/, '') || '/';

  return withoutLocale === '/' ? `/${locale}` : `/${locale}${withoutLocale}`;
};

export const getOtherLocale = (locale: Locale): Locale => (locale === 'ko' ? 'en' : 'ko');
