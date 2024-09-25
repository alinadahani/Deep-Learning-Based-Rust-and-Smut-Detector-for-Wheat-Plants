// index.js

// Function to handle file selection
function handleFileSelect(event) {
    const input = event.target;
    if (input.files && input.files.length > 0) {
        const fileName = input.files[0].name;
        updateFileNameDisplay(fileName);
        hideUploadInstructions();
    }
}

// Function to update file name display
function updateFileNameDisplay(fileName) {
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    fileNameDisplay.innerText = fileName;
}

// Function to hide upload instructions after file upload
function hideUploadInstructions() {
    const uploadBox = document.getElementById('uploadBox');
    const uploadInstructions = uploadBox.querySelector('p');
    uploadInstructions.style.display = 'none';
}
