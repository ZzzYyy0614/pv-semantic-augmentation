// Optional developer utility: npm install sharp, then node scripts/render_assets.cjs.
// A custom Sharp module path may be supplied as the first argument.
const path = require("node:path");
const sharp = require(process.argv[2] || "sharp");
const assets = path.resolve(__dirname, "../assets");
sharp(path.join(assets, "framework.svg"))
  .png()
  .toFile(path.join(assets, "framework.png"))
  .then(({ width, height }) => process.stdout.write(`Rendered framework.png (${width} x ${height})\n`))
  .catch((error) => { process.stderr.write(error.message + "\n"); process.exitCode = 1; });
