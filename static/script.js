let selectedFiles = [];

const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const selectedFileInfo = document.getElementById('selectedFileInfo');
const messageDiv = document.getElementById('message');

// Nghe sự kiện chọn file từ input
fileInput.addEventListener('change', handleFiles);

// Kéo thả
dropArea.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropArea.classList.add('dragging');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragging');
});

dropArea.addEventListener('drop', (event) => {
    event.preventDefault();
    dropArea.classList.remove('dragging');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        selectedFiles = Array.from(files);
        displaySelectedInfo();
    }
});

function handleFiles(event) {
    selectedFiles = Array.from(event.target.files);
    displaySelectedInfo();
}

function displaySelectedInfo() {
    if (selectedFiles.length === 1) {
        selectedFileInfo.innerText = `File selected: ${selectedFiles[0].name}`;
    } else {
        selectedFileInfo.innerText = "No file selected.";
    }
}

// Xử lý sự kiện khi nhấn nút tải lên
document.getElementById('uploadBtn').addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        messageDiv.innerText = "Please select or drop a file.";
        return;
    }

    const formData = new FormData();
    selectedFiles.forEach(file => formData.append('file', file));

    // Hiển thị trạng thái "Đang xử lý..."
    messageDiv.innerText = "Loading...";

    try {
        const response = await fetch('/predict-image/', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        // Hiển thị thông báo trả về từ server
        if (response.ok) {
            messageDiv.innerText = result.info;
        } else {
            messageDiv.innerText = "An error occurred while processing the image.";
        }
    } catch (error) {
        console.error("Error:", error);
        messageDiv.innerText = "Failed to connect to the server.";
    }
});
