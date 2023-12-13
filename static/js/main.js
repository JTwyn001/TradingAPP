$(document).ready(function() {

    $('.text').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "bounceIn",
        },
        out: {
            effect: "bounceOut",
        },
    });

    // JavaScript function to update button text
    function updateButtonText(text) {
        var button = document.getElementById('dropdownMenuButton1');
        button.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';
    };

    // Handling Q/A Submission
    document.getElementById('submitQuery').addEventListener('click', function() {
        var userInput = document.getElementById('userInput').value;
        fetch('/process_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'user_input': userInput }),
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById('outputSection').innerHTML = JSON.stringify(data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    });

});
