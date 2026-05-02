const chatForm = document.getElementById('chat-form');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const messageType = document.getElementById('message-type')
const clearButton = document.getElementById('clear-button');
const contextModal = document.getElementById("contextModal");
const contextButton = document.getElementById('context-button');
const closeContextModal = document.getElementById('closeContextModal');
const loadContextInput = document.getElementById('loadContextInput');
const loadButton = document.getElementById('load-button');
const pasteContextButton = document.getElementById('paste-context-button');

const genreDiv = document.getElementById('genre-div');
const mangaNameInput = document.getElementById('manga-name-text-input');
const chapterCountInput = document.getElementById('chapter-count-range');

const saveButton = document.getElementById('save-button');
//Image Modal
const imageModal = document.getElementById('imageModal');
const imageModalImg = document.getElementById('imageModalImg');
const lastImageButton = document.getElementById('lastImageButton');
const nextImageButton = document.getElementById('nextImageButton');

//Context text areas
const synopsisArea = document.getElementById('synopsisTextArea');
const characterArea = document.getElementById('characterTextArea');
const outlineArea = document.getElementById('outlineTextArea');
const panelArea = document.getElementById('panelTextArea');
const promptArea = document.getElementById('promptTextArea');


let currImgIndex = 0;
let imgWidth = 576;
let imgHeight = 1024;

//Image loading promise so it can be awaited
const loadImage = src =>
  new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
});

const SENDER_ID = Object.freeze({
    HUMAN: Symbol("user"),
    BOT: Symbol("bot"),
    ERROR: Symbol("err"),
})

//TODO: Add manga title input and chapter number selector.

const genres = ["Romance", "Comedy", "Drama", "Action", "Fantasy", "School", "Supernatural",
  "Slice of Life", "Adventure", "Sci-Fi", "Mystery", "Historical", "Horror"];

genres.forEach(element => {
  let label = document.createElement("label");
  label.className = "checkbox-container";
  let checkbox = document.createElement("input");
  checkbox.id = element;
  checkbox.name = element;
  checkbox.value = element;
  checkbox.type = "checkbox";
  let div = document.createElement("div");
  div.className = "checkbox-div";
  div.innerHTML = element;
  label.appendChild(checkbox);
  label.appendChild(div);
  genreDiv.appendChild(label);
});

const inputPlaceholders = {
  "manga": "Enter manga description...",
  "synopsis": "Enter manga description...",
  "characters": "Enter manga synopsis or use synopsis in context...",
  "outline": "Enter manga synopsis or use synopsis in context...",
  "panels": "Enter outline or use outline in context...",
  "prompts": "Enter panel information or use panels in context...",
  "image": "Enter prompt for image...",
  "image_dummy": "Enter prompt for image...",
  "help": "What would you like help with?",
}

function idToString(id){
  switch(id){
    case SENDER_ID.HUMAN:
      return "user";
    case SENDER_ID.BOT:
      return "bot";
    case SENDER_ID.ERROR:
      return "err";
  }
}

document.getElementById("user-input").addEventListener("keypress", e => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!e.repeat) {
            const newEvent = new Event("submit", {cancelable: true});
            e.target.form.dispatchEvent(newEvent);
        }
    }
});

function addMessage(text, sender){
  const bubble = document.createElement('div');
  const idName = idToString(sender);
  bubble.className = `message ${idName}`;
  console.log("Adding message: " + text + " from sender: " + idName);
  if(text!="")
    text = text.replace(/\n/g, '<br>');
  bubble.innerHTML = text;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

function deleteMessage(bubble){
  chatMessages.removeChild(bubble);
}

function clearMessages(){
  chatMessages.innerHTML = '';
}

function toggleInputDisable(value){
  userInput.disabled = value;
  clearButton.disabled = value;
}

function getBotReplyStream(message, type) {
  console.log("Getting bot reply for message: " + message);
  message = encodeURIComponent(message);

  toggleInputDisable(true);
  let response = addMessage("", SENDER_ID.BOT);
  const evtSource = new EventSource('http://127.0.0.1:4500/' + type + '_stream?param1=' + message);

  evtSource.onmessage = function(event) {
    console.log("Received event: " + event.data);
    let data = JSON.parse(event.data);
    let text = data.text.replace(/\n/g, '<br>');
    response.innerHTML += text;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  };

  evtSource.onerror = function(err) {
    console.error("Error: ", err);
    toggleInputDisable(false);
    evtSource.close();
  }

  evtSource.addEventListener('end', function(event) {
    console.log("Received end event: " + event.data);
    toggleInputDisable(false);
    evtSource.close();
  });
}

function getLines(ctx, text, maxWidth) {
    let words = text.split(" ");
    let lines = [];
    let currentLine = words[0];

    for (let i = 1; i < words.length; i++) {
        let word = words[i];
        let width = ctx.measureText(currentLine + " " + word).width;
        
        if (width < maxWidth) {
            currentLine += " " + word;
        } else {
            lines.push(currentLine);
            currentLine = word;
        }
    }
    lines.push(currentLine);
    
    return lines;
}

function drawSpeechBubble(ctx, text, pos, maxWidth, maxHeight){
  let padding = 10;
  let bottomPadding = 40;
  let fontSize = 30;

  ctx.font = fontSize + "px Arial";
  ctx.textAlign = "start";
  ctx.textBaseline = "hanging";
  let textWidth = ctx.measureText(text).width;
  let lines = getLines(ctx, text, maxWidth - padding * 4);
  let textHeight = lines.length * fontSize + padding * 2;
  pos.y = maxHeight - textHeight - bottomPadding;
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = "4";
  ctx.strokeStyle = "black"
  ctx.fillRect(pos.x + padding, pos.y, maxWidth - padding * 2, textHeight);
  ctx.strokeRect(pos.x + padding, pos.y, maxWidth - padding * 2, textHeight);

  lines.forEach(line => {
    ctx.fillStyle = "#000000";
    ctx.fillText(line, pos.x + padding * 2, pos.y + padding);
    pos.y += fontSize;
  });
}

const b64toBlob = (b64Data, contentType='', sliceSize=512) => {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }
    
  const blob = new Blob(byteArrays, {type: contentType});
  return blob;
}

async function generateImages(message, type){  
  toggleInputDisable(true);
  let bubble = addMessage("", SENDER_ID.BOT);
  let returnData = ""

  let prompts = message.split(/\*\*Panel\s[0-9]+\*\*\sPrompt:\s/i);
  prompts.splice(0, 1);
  let conversation = panelArea.value.split(/\*\*Panel\s[0-9]+\*\*\s/i).join("").split("*");
  conversation.splice(0, 1);
  let convos = [];
  let index = -1;
  conversation.forEach(c => {
    if(c.startsWith("Scene Description")){
      convos.push("");
      index++;
    }
    else
      convos[index] += c + "\n";
  });  

  
  let imgContainer = document.createElement("div");
  imgContainer.className = "img-container";
  bubble.appendChild(imgContainer);  
  
  let canvas = document.createElement("canvas");
  canvas.width = imgWidth;
  canvas.height = imgHeight;
  let ctx = canvas.getContext("2d");

  try{
    for(let i = 0; i < prompts.length; i++){
      let prompt = prompts[i] + " in the style of manga black and white";
      let convo = convos[i];
      let url = 'http://127.0.0.1:4500/' + type;
      let args = {"param1": prompt, "width": imgWidth, "height": imgHeight};
      
      message = encodeURIComponent(message);
      await $.post(url, args,
      function(data, textStatus, jqXHR) {
        returnData = data;
      
        response = returnData.replace(/\n/g, "<br>");
      }
      
      );
      let src = 'data:image/png;base64,' + returnData;      
      await loadImage(src).then(img => {
        img.width = imgWidth;
        img.height = imgHeight;

        ctx.drawImage(img, 0, 0, imgWidth, imgHeight);
        if(convo.trim() != "")
          drawSpeechBubble(ctx, convo, {x: 0, y: 0}, canvas.width, canvas.height);
        img.src = canvas.toDataURL("image/png");
        imgContainer.appendChild(img);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      });

    }
  }
  catch(err){
    deleteMessage(bubble);
    console.log(err);
    
    addMessage("Couldn't get response from bot.", SENDER_ID.ERROR);
    return null;
  }
  
  //Add button to download the images presumably.
  let downloadButton = document.createElement("button");
  downloadButton.className = "bot-message-button";
  downloadButton.title = "Download Panels";
  const buttonImage = document.createElement("img");
  buttonImage.src = "src/assets/images/download.png";
  downloadButton.appendChild(buttonImage);

  
  downloadButton.addEventListener('click', async function downloadPanels(){
    let images = imgContainer.getElementsByTagName("img");
    let blobs = [];
    for(let i = 0; i < images.length; i++){
      let image = images.item(i);
      let src = image.src;
      let contentType = src.split("data:")[1].split(";base64,")[0];
      let b64Data = src.split("base64,")[1];
      let blob = b64toBlob(b64Data, contentType);
      blobs.push(blob);
    }
    console.log(blobs);
    
    const zip = new JSZip();
    blobs.forEach((blob, index) => {
      zip.file('panel' + (index+1) + '.png', blob);
    });

    const res = await zip.generateAsync({type: 'blob'});
    DownloadFile(res, "panels.zip");
    
    this.blur();
  });
  bubble.innerHTML += "<hr>";
  bubble.appendChild(downloadButton);

  
  toggleInputDisable(false);
  return bubble;
}

async function getBotReply(message, type, generateBubble = true){
  console.log("Getting bot reply for message: " + message);
  // message = encodeURIComponent(message);


  toggleInputDisable(true);
  let response = "...";
  let returnData = "...";
  let args = {};
  let bubble = null;
  if(generateBubble)
    bubble = addMessage(response, SENDER_ID.BOT);
  let url = 'http://127.0.0.1:4500/' + type + '?';
  if(message)
    args["param1"] = message;


  let checkboxes = genreDiv.getElementsByTagName("input");
  for(let i = 0; i < checkboxes.length; i++){
    let checkbox = checkboxes.item(i);
    if(checkbox.checked){
      args["genres"] += checkbox.value + ", ";
    }
  }


  switch(type){
    case 'manga':
      args["scenes"] = chapterCountInput.value;
      break;
    case "panels":
      args["synopsis"] = synopsisArea.value;
      args["characters"] = characterArea.value;
      args["outline"] = outlineArea.value;
      break;
    case "characters":
      args["synopsis"] = synopsisArea.value;
      break;
    case "outline":
      args["synopsis"] = synopsisArea.value;
      args["characters"] = characterArea.value;
      args["scenes"] = chapterCountInput.value;
      break;
    case "prompts":
      args["characters"] = characterArea.value;
      break;
  }
  

  try{
    await $.post(url, args, function(data){
      returnData = data;
      if(type != "manga")
          response = returnData.replace(/\n/g, "<br>");
    })
    
    // else{
    //   await $.getJSON(url,
    //     function(data, textStatus, jqXHR) {
    //       returnData = data;
        
    //       if(type != "manga")
    //         response = returnData.replace(/\n/g, "<br>");

    //     }
    //   );
    // }

    // if(type == "manga"){
    //   synopsisArea.value = returnData.synopsis;
    //   characterArea.value = returnData.characters;
    //   outlineArea.value = returnData.outline;
    //   panelArea.value = returnData.panels;
    //   promptArea.value = returnData.prompts;
    //   deleteMessage(bubble);


    //   generateImages(returnData.prompts, "image")
    //   return "";
    // }



    if(generateBubble){
      bubble.innerHTML = response;
      let copy_button = document.createElement("button");
      copy_button.className = "bot-message-button";
      copy_button.id = type;
      copy_button.title = "Copy to context"
      copy_button.addEventListener('click', function copyToContext(){
        switch(this.id){
          case "synopsis":
            synopsisArea.value = returnData;
            break;
          case "characters":
            characterArea.value = returnData;
            break;
          case "outline":
            outlineArea.value = returnData;
            break;
          case "panels":
            panelArea.value = returnData;
            break;
          case "prompts":
            promptArea.value = returnData;
            break;
          default:
            navigator.clipboard.writeText(returnData).then(function(){
                console.log("Successfully copied to clipboard!");
            }, function(err){
                console.log("Failed to copy to clipboard." + err);
            });
            break;
        }
        this.blur();
      });
      let button_img = document.createElement("img");
      button_img.src = "src/assets/images/copy.png";
      copy_button.appendChild(button_img);
  
      bubble.innerHTML += "<hr>";
      bubble.appendChild(copy_button);
    }
  }
  catch(error){
    if(generateBubble)
      deleteMessage(bubble);
    console.log(error);
    
    addMessage("Couldn't get response from bot.", SENDER_ID.ERROR);
    return "";
  }



  toggleInputDisable(false);
  return returnData;
}

function getBotReplyDummy(message){
  addMessage("You wrote: " + message, SENDER_ID.BOT);
}

function addImgClickListeners(bubble){
  if(bubble == null) return;
  let imgContainer = bubble.getElementsByClassName("img-container").item(0);
  let imgs = imgContainer.getElementsByTagName("img");
  for(let i = 0; i < imgs.length; i++){
    let img = imgs.item(i);    

    img.addEventListener('click', function showImageModal(){
      imageModal.style.display = "flex";
      imageModalImg.src = img.src;
      imageModalImg.width = imgWidth/1.2;
      imageModalImg.height = imgHeight/1.2;
      currImgIndex = Array.prototype.indexOf.call(imgContainer.children, img);
      lastImageButton.addEventListener('click', function lastImage(){
        if(currImgIndex == 0) return;
        currImgIndex--;
        let imgs = imgContainer.getElementsByTagName("img");
                    
        imageModalImg.src = imgs.item(currImgIndex).src;
      });
      nextImageButton.addEventListener('click', function lastImage(){
        if(currImgIndex >= imgs.length - 1) return;
        currImgIndex++;
                    
        imageModalImg.src = imgs.item(currImgIndex).src;
      });
    });
  }
}

chatForm.addEventListener('submit', async function (event) {
  event.preventDefault();

  const type = messageType.value;
  const message = userInput.value.trim();
  
  if (!message) {
    return;
  }

  addMessage(message, SENDER_ID.HUMAN);
  userInput.value = '';
  userInput.focus();
  if(type == "image" || type == "image_dummy"){
    let bubble = await generateImages(message, type);
    addImgClickListeners(bubble);
    
  }
  else if(type == "manga"){
    let bubble = addMessage("Generating synopsis from prompt...", SENDER_ID.BOT);
    let data = "";
    // data = await getBotReply(message, "synopsis", false);
    // if(data == ""){
    //   deleteMessage(bubble);
    //   return;
    // }
    // synopsisArea.value = data;
    bubble.innerHTML += "<br><br>Generating characters...";
    data = await getBotReply(synopsisArea.value, "characters", false);
    if(data == ""){
      deleteMessage(bubble);
      return;
    }
    characterArea.value = data;
    bubble.innerHTML += "<br><br>Generating outline...";
    data = await getBotReply(synopsisArea.value, "outline", false);
    if(data == ""){
      deleteMessage(bubble);
      return;
    }
    outlineArea.value = data;
    bubble.innerHTML += "<br><br>Generating story panels...";
    data = await getBotReply(outlineArea.value, "panels", false);
    if(data == ""){
      deleteMessage(bubble);
      return;
    }
    panelArea.value = data;
    bubble.innerHTML += "<br><br>Generating image prompts...";
    data = await getBotReply(panelArea.value, "prompts", false);
    if(data == ""){
      deleteMessage(bubble);
      return;
    }
    promptArea.value = data;
    deleteMessage(bubble);
    bubble = await generateImages(message, type);
    addImgClickListeners(bubble);
  }
  else
    getBotReply(message, type);

});

clearButton.addEventListener('click', function () {
  clearMessages();
});

contextButton.addEventListener('click', function (){
  contextModal.style.display = "flex";
});

closeContextModal.onclick = function() {
  contextModal.style.display = "none";
};

window.onclick = function(event) {
  if (event.target == contextModal || event.target == imageModal) {
    contextModal.style.display = "none";
    imageModal.style.display = "none";
  }
};

loadButton.addEventListener('click', function(e){
  if(loadContextInput)
    loadContextInput.click()

  e.preventDefault();
});

loadContextInput.addEventListener("change", async function (){
  const fileList = this.files;
  if(fileList.length <= 0) return;

  const file = fileList[0];
  let jsonContent = await loadJSONFile(URL.createObjectURL(file));
  //Actually load the content
  synopsisArea.value = jsonContent["synopsis"];
  characterArea.value = jsonContent["characters"];
  outlineArea.value = jsonContent["outline"];
  panelArea.value = jsonContent["panels"];
  promptArea.value = jsonContent["prompts"];
  mangaNameInput.value = jsonContent["name"];
  chapterCountInput.value = jsonContent["chapters"];
  let genres = jsonContent["genres"];
  let checkboxes = genreDiv.getElementsByTagName("input");
  for(let i = 0; i < checkboxes.length; i++){
      let checkbox = checkboxes.item(i);
      if(genres.includes(checkbox.value)){
        checkbox.checked = true;
      }
    }

});


async function loadJSONFile(path){
    let response = await fetch(path);
    return await response.json();
};

saveButton.addEventListener('click', function(){
    let contextInfo = document.getElementById('context-modal-body').getElementsByTagName("textarea");
    let genres = "";
    let checkboxes = genreDiv.getElementsByTagName("input");
    for(let i = 0; i < checkboxes.length; i++){
      let checkbox = checkboxes.item(i);
      if(checkbox.checked){
        genres += checkbox.value + ", ";
      }
    }

    //TODO: add manga name and chapter number information.
    let saveInfo = {
      "name": mangaNameInput.value,
      "chapters": chapterCountInput.value,
      "synopsis": contextInfo[0].value,
      "characters": contextInfo[1].value,
      "outline": contextInfo[2].value,
      "panels": contextInfo[3].value,
      "genres": genres,
      "prompts": contextInfo[4].value,
    };

    //TODO: set the file name to name of manga generated.
    let blobData = new Blob([JSON.stringify(saveInfo, null, 2)], {
      type: "text/plain"
    });
    DownloadFile(blobData, saveInfo["name"] + ".json");
});

function DownloadFile(data, fileName){
    const a = document.createElement("a");
    a.href = URL.createObjectURL(data);
    a.setAttribute("download", fileName);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
};

messageType.addEventListener("change", function(){
  const value = this.value;
  userInput.placeholder = inputPlaceholders[value];
});

pasteContextButton.addEventListener('click', function(){
  switch(messageType.value){
    case "characters":
    case "outline":
      userInput.value = synopsisArea.value.trim();
      break;
    case "panels":
      userInput.value = outlineArea.value.trim();
      break;
    case "prompts":
      userInput.value = panelArea.value.trim();
      break;
    case "image_dummy": 
    case "image":
      userInput.value = promptArea.value.trim();
      break;
  }
});