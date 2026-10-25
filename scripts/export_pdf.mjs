// Optional PDF export through an already installed sandbox-capable Chromium.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const {chromium}=await import(process.env.LEARN_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const browser=await chromium.launch({executablePath:process.env.LEARN_CHROMIUM||chromium.executablePath(),headless:true,chromiumSandbox:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}});
 await page.goto(pathToFileURL(path.join(root,'site/book.html')).href);
 await page.evaluate(()=>document.fonts.ready);await page.emulateMedia({media:'print'});
 fs.mkdirSync(path.join(root,'output/pdf'),{recursive:true});
 await page.pdf({path:path.join(root,'output/pdf/Learn-DeepInfer.pdf'),format:'A4',preferCSSPageSize:true,printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-family:Arial;font-size:8px;width:100%;text-align:center;color:#64756e">Learn-DeepInfer · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',tagged:true,outline:true});
 console.log('PDF exported with sandbox enabled. Browser closed on completion.');
}finally{await browser.close();}
