document.getElementById('uploadImage').addEventListener('click', async () => {
    const githubUsername = prompt("Enter your GitHub username:");
    if (!githubUsername) return alert("GitHub username is required.");

    try {
        await createFork(githubUsername);
        const imageFile = await uploadImage();
        await uploadToRepo(githubUsername, imageFile);
        await runLabelScript(githubUsername);
        await uploadJsonFile(githubUsername);
        alert("Process completed and pull request created.");
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred during the process.");
    }
});

async function createFork(githubUsername) {
    const response = await fetch(`https://api.github.com/repos/A5pir1n/Pixel-Labeler/forks`, {
        method: 'POST',
        headers: {
            'Authorization': `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`,
            'Accept': 'application/vnd.github.v3+json'
        }
    });

    if (!response.ok) throw new Error("Failed to create fork.");
}

async function uploadImage() {
    return new Promise((resolve, reject) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = () => {
            const file = input.files[0];
            if (file) resolve(file);
            else reject("No file selected.");
        };
        input.click();
    });
}

async function uploadToRepo(githubUsername, imageFile) {
    const reader = new FileReader();
    reader.readAsDataURL(imageFile);

    reader.onload = async () => {
        const base64Image = reader.result.split(',')[1];
        const content = btoa(base64Image);

        const response = await fetch(`https://api.github.com/repos/${githubUsername}/Pixel-Labeler/contents/images/${imageFile.name}`, {
            method: 'PUT',
            headers: {
                'Authorization': `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`,
                'Accept': 'application/vnd.github.v3+json'
            },
            body: JSON.stringify({
                message: "Upload image",
                content: content
            })
        });

        if (!response.ok) throw new Error("Failed to upload image.");
    };
}

async function runLabelScript(githubUsername) {
    const response = await fetch(`https://api.github.com/repos/${githubUsername}/Pixel-Labeler/actions/workflows/label.yml/dispatches`, {
        method: 'POST',
        headers: {
            'Authorization': `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`,
            'Accept': 'application/vnd.github.v3+json'
        },
        body: JSON.stringify({
            ref: "main"
        })
    });

    if (!response.ok) throw new Error("Failed to run label script.");
}

async function uploadJsonFile(githubUsername) {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = async () => {
        const file = input.files[0];
        if (!file) throw new Error("No JSON file selected.");

        const reader = new FileReader();
        reader.readAsText(file);
        reader.onload = async () => {
            const content = btoa(reader.result);

            const response = await fetch(`https://api.github.com/repos/${githubUsername}/Pixel-Labeler/contents/labels/${file.name}`, {
                method: 'PUT',
                headers: {
                    'Authorization': `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`,
                    'Accept': 'application/vnd.github.v3+json'
                },
                body: JSON.stringify({
                    message: "Upload label JSON",
                    content: content
                })
            });

            if (!response.ok) throw new Error("Failed to upload JSON file.");

            await createPullRequest(githubUsername);
        };
    };
    input.click();
}

async function createPullRequest(githubUsername) {
    const response = await fetch(`https://api.github.com/repos/A5pir1n/Pixel-Labeler/pulls`, {
        method: 'POST',
        headers: {
            'Authorization': `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`,
            'Accept': 'application/vnd.github.v3+json'
        },
        body: JSON.stringify({
            title: "New Label Submission",
            head: `${githubUsername}:main`,
            base: "main",
            body: "New label data submitted."
        })
    });

    if (!response.ok) throw new Error("Failed to create pull request.");
}
