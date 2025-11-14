// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // TMDB poster base domain
    domains: ['image.tmdb.org'],
  },
};

module.exports = nextConfig;
