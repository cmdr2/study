// Bookmarklet version:
// javascript:(function(){const OPENAI_HOSTNAME='http://localhost:1234',MODEL_NAME='Llama-3.1-8B-Lexi-Uncensored-V2-GGUF';(async function(){const t=window.getSelection()||document.selection.createRange().text;if(t){try{const r=await fetch(`${OPENAI_HOSTNAME}/v1/chat/completions`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:MODEL_NAME,messages:[{role:'user',content:`Summarize this in 200 words or less. Skip filler words, articles, and unnecessary words. Respond with the summary directly, don't write a preceding lead-up:\n\n${t}`}]})});const d=await r.json();alert(d.choices[0].message.content)}catch(e){console.error('Error:',e);}}else{alert('No text selected')};})();})();

// Global Constants
const OPENAI_HOSTNAME = 'http://localhost:1234';
const MODEL_NAME = 'Llama-3.1-8B-Lexi-Uncensored-V2-GGUF';

function getSelectedText() {
    var text = '';
    if (typeof window.getSelection != 'undefined') { // webkit browsers (Chrome, Safari)
        text = window.getSelection().toString();
    } else if (typeof document.selection != 'undefined') { // IE
        text = document.selection.createRange().text;
    }
    return text;
}

async function summarize(textToSummarize, expectedWords) {
    console.log(textToSummarize);
    
    if (textToSummarize) {
        const apiEndpoint = `${OPENAI_HOSTNAME}/v1/chat/completions`;
        try {
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: MODEL_NAME,
                    messages: [
                        {
                            role: 'user',
                            content: `Summarize this in ${expectedWords} words or less. Skip filler words, articles, and unnecessary words. Respond with the summary directly, don't write a preceding lead-up:\n\n${textToSummarize}`
                        }
                    ]
                })
            });
            
            const data = await response.json();
            return data.choices[0].message.content;
        } catch (error) {
            console.error('Error:', error);
            return null; // Return null to indicate an error occurred
        }
    } else {
        console.log('No text selected');
        return null; // Return null to indicate no text was selected
    }
}

async function main() {
    const selectedText = getSelectedText();
    
    if (!selectedText) {
        alert('No text selected');
        return;
    }
    
    const summary = await summarize(selectedText, 200);
    alert(summary);
}

main();
