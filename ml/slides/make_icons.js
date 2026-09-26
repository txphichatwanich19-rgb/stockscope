const React = require('react');
const ReactDOMServer = require('react-dom/server');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const {
  FiTrendingUp, FiDatabase, FiCpu, FiShield, FiActivity,
  FiBarChart2, FiClock, FiTarget, FiAlertTriangle, FiArrowRight,
  FiCheckCircle, FiXCircle, FiLayers, FiZap, FiGitBranch,
} = require('react-icons/fi');

const ICONS = {
  'trending-up': FiTrendingUp,
  'database': FiDatabase,
  'cpu': FiCpu,
  'shield': FiShield,
  'activity': FiActivity,
  'bar-chart': FiBarChart2,
  'clock': FiClock,
  'target': FiTarget,
  'alert-triangle': FiAlertTriangle,
  'arrow-right': FiArrowRight,
  'check-circle': FiCheckCircle,
  'x-circle': FiXCircle,
  'layers': FiLayers,
  'zap': FiZap,
  'git-branch': FiGitBranch,
};

const OUT_DIR = path.join(__dirname, 'icons');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR);

async function run() {
  const colors = {
    'lime': '#A3E635',
    'white': '#F5F5F5',
    'dark': '#0A0E14',
    'coral': '#FF6B6B',
    'muted': '#9AA5B1',
  };
  for (const [name, Comp] of Object.entries(ICONS)) {
    for (const [colorName, hex] of Object.entries(colors)) {
      const svg = ReactDOMServer.renderToStaticMarkup(
        React.createElement(Comp, { color: hex, size: 256, strokeWidth: 1.6 })
      );
      const fullSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 24 24">${svg.replace(/<svg[^>]*>|<\/svg>/g, '')}</svg>`;
      const outPath = path.join(OUT_DIR, `${name}-${colorName}.png`);
      await sharp(Buffer.from(fullSvg)).resize(256, 256).png().toFile(outPath);
    }
  }
  console.log('Icons written to', OUT_DIR);
}

run().catch(e => { console.error(e); process.exit(1); });
