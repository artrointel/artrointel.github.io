import type { Locale } from './i18n';

const labels = {
  en: {
    about: 'About',
    resume: 'Resume',
    lab: 'Lab',
    blog: 'Blog',
    sitemap: 'Sitemap',
    connect: 'Connect',
    email: 'Email',
    rights: 'All rights reserved.',
  },
  ko: {
    about: '소개',
    resume: '이력서',
    lab: '프로젝트',
    blog: '블로그',
    sitemap: '사이트맵',
    connect: '연락처',
    email: '이메일',
    rights: '모든 권리 보유.',
  },
} as const;

const localPath = (locale: Locale, path = '') => `/${locale}${path}`;

export const getHeaderData = (locale: Locale = 'en') => ({
  links: [
    {
      text: labels[locale].about,
      href: localPath(locale, '/about'),
    },
    {
      text: labels[locale].resume,
      href: localPath(locale, '/resume'),
    },
    {
      text: labels[locale].lab,
      href: localPath(locale, '/lab'),
    },
    {
      text: labels[locale].blog,
      href: localPath(locale, '/blog'),
    },
  ],
  actions: [], // Removed Download CV button
});

export const getFooterData = (locale: Locale = 'en') => ({
  links: [
    {
      title: labels[locale].sitemap,
      links: [
        { text: labels[locale].about, href: localPath(locale, '/about') },
        { text: labels[locale].resume, href: localPath(locale, '/resume') },
        { text: labels[locale].lab, href: localPath(locale, '/lab') },
        { text: labels[locale].blog, href: localPath(locale, '/blog') },
      ],
    },
    {
      title: labels[locale].connect,
      links: [
        { text: 'GitHub', href: 'https://github.com/artrointel' },
        { text: labels[locale].email, href: 'mailto:artrointel@gmail.com' },
      ],
    },
  ],
  secondaryLinks: [],
  socialLinks: [
    { ariaLabel: 'Github', icon: 'tabler:brand-github', href: 'https://github.com/artrointel' },
    { ariaLabel: 'Email', icon: 'tabler:mail', href: 'mailto:artrointel@gmail.com' },
  ],
  footNote: `
    <a class="text-blue-600 underline dark:text-muted" href="https://github.com/artrointel">Artrointel</a>. ${labels[locale].rights}
  `,
});

export const headerData = getHeaderData('en');
export const footerData = getFooterData('en');
