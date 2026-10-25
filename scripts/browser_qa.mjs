// Uses the environment's already qualified sandboxed Chromium, with no install.
const { chromium } = await import(process.env.LEARN_PLAYWRIGHT_MODULE || 'playwright');
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
fs.mkdirSync(path.join(root,'reports'),{recursive:true});
const bin=process.env.LEARN_CHROMIUM || chromium.executablePath();
const evidence={sandbox:true,errors:[],checks:[],externalRequests:[]};
let browser;
const check=(name,value)=>{evidence.checks.push({name,passed:!!value});if(!value)throw new Error(name)};
try{
 browser=await chromium.launch({executablePath:bin,headless:true,chromiumSandbox:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000}});const page=await context.newPage();
 page.on('pageerror',e=>evidence.errors.push(String(e)));page.on('request',r=>{if(/^https?:/.test(r.url()))evidence.externalRequests.push(r.url())});
 const url=name=>pathToFileURL(path.join(root,'site',name)).href;
 await page.goto(url('index.html'));await page.evaluate(()=>document.fonts.ready);
 check('15 chapter cards',await page.locator('.chapter-card').count()===15);
 check('desktop no horizontal overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:path.join(root,'reports/home-desktop.png'),fullPage:true});
 await page.locator('.chapter-card[href="01-loop.html"]').click();
 await page.locator('#search').fill('两轮');check('offline search returns results',await page.locator('#search-results a').count()>0);
 await page.locator('#search').fill('unlikely-nonexistent-query');check('empty search handled',(await page.locator('#search-results').innerText()).includes('0 个章节'));
 await page.locator('#search').fill('');
 const completion=page.locator('.complete');const before=await completion.getAttribute('aria-pressed');await completion.click();check('progress toggles',await completion.getAttribute('aria-pressed')!==before);await page.reload();check('progress persists',await completion.getAttribute('aria-pressed')!==before);await completion.click();
 await page.locator('#theme').click();check('dark theme active',await page.locator('body').getAttribute('data-theme')==='dark');await page.reload();check('theme survives reload',await page.locator('body').getAttribute('data-theme')==='dark');await page.locator('#theme').click();
 await page.goto(url('09-evidence.html'));check('trace cases loaded',await page.locator('#trace-case option').count()===4);
 const first=await page.locator('#trace-status').innerText();await page.locator('#trace-next').click();check('trace next changes state',await page.locator('#trace-status').innerText()!==first);
 await page.locator('#trace-position').evaluate(el=>{el.value=el.max;el.dispatchEvent(new Event('input',{bubbles:true}))});check('final trace verified',(await page.locator('#trace-event').innerText()).includes('complete'));
 check('final trace displays true/false', (await page.locator('#trace-requests').innerText()).includes('true')&&(await page.locator('#trace-requests').innerText()).includes('false'));
 await page.locator('#trace-case').selectOption('03_block_lifecycle');check('changing trace resets',await page.locator('#trace-position').inputValue()==='0');
 await page.locator('#trace-case').selectOption('09_metrics');await page.locator('#trace-position').evaluate(el=>{el.value=25;el.dispatchEvent(new Event('input',{bubbles:true}))});
 await page.locator('.trace-lab').screenshot({path:path.join(root,'reports/trace-desktop.png')});
 await page.setViewportSize({width:390,height:844});await page.goto(url('index.html'));check('mobile no horizontal overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.locator('#menu').click();check('mobile menu opens',await page.locator('#menu').getAttribute('aria-expanded')==='true');await page.keyboard.press('Escape');check('Escape closes mobile menu',await page.locator('#menu').getAttribute('aria-expanded')==='false');await page.screenshot({path:path.join(root,'reports/home-mobile.png'),fullPage:true});
 await page.goto(url('06-async.html'));check('mobile lesson no horizontal overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:path.join(root,'reports/lesson-mobile.png'),fullPage:true});
 await page.goto(url('book.html'));await page.evaluate(()=>document.fonts.ready);check('book full chapter count',await page.locator('.book-chapter').count()===15);check('book trace functional',await page.locator('#trace-case option').count()===4);
 check('no browser errors',evidence.errors.length===0);check('no external assets',evidence.externalRequests.length===0);
 if(process.argv.includes('--pdf')){
   fs.mkdirSync(path.join(root,'output/pdf'),{recursive:true});
   await page.emulateMedia({media:'print'});
   await page.pdf({path:path.join(root,'output/pdf/Learn-DeepInfer.pdf'),format:'A4',preferCSSPageSize:true,printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="font-family:Arial;font-size:8px;width:100%;text-align:center;color:#64756e">Learn-DeepInfer · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',tagged:true,outline:true});
   evidence.pdfBytes=fs.statSync(path.join(root,'output/pdf/Learn-DeepInfer.pdf')).size;
 }
} catch(e){evidence.failure=String(e);process.exitCode=1;}finally{if(browser)await browser.close();fs.writeFileSync(path.join(root,'reports/browser-qa.json'),JSON.stringify(evidence,null,2));console.log(JSON.stringify(evidence,null,2));}
