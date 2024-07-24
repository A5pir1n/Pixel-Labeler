document.getElementById('uploadImage').addEventListener('click', function() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async (event) => {
        const file = event.target.files[0];
        if (file) {
            await uploadFile(file);
        }
    };
    input.click();
});

document.getElementById('labelExistingImages').addEventListener('click', function() {
    // Redirect to a page or function to label existing images
    window.location.href = 'label_existing.html';
});

document.getElementById('challengeLabeling').addEventListener('click', function() {
    // Redirect to a page or function to challenge existing labeling
    window.location.href = 'challenge_labeling.html';
});

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('https://api.github.com/repos/yourusername/my-image-labeler/contents/user_uploads/' + file.name, {
            method: 'PUT',
            headers: {
                'Authorization': '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',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: 'Upload image ' + file.name,
                content: await fileToBase64(file),
                branch: 'main'
            })
        });

        if (response.ok) {
            alert('File uploaded successfully!');
        } else {
            const responseData = await response.json();
            console.error('Error response from GitHub:', responseData);
            alert('Failed to upload file: ' + responseData.message);
        }
    } catch (error) {
        console.error('Error uploading file:', error);
        alert('Failed to upload file.');
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.onerror = error => reject(error);
    });
}
