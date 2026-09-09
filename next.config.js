/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: '/RoadScan_AI',
  assetPrefix: '/RoadScan_AI/',
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
