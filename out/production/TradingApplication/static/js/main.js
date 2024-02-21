// JavaScript function to update button text
function updateButtonText(text) {
    var button = document.getElementById('dropdownMenuButton1');
    button.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';
};

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

    document.addEventListener('DOMContentLoaded', function() {
        // Query all dropdown items
        var items = document.querySelectorAll('.dropdown-item');
        items.forEach(function(item) {
            item.addEventListener('click', function() {
                var text = this.innerText; // 'this' refers to the item clicked
                updateButtonText(text);
            });
        });
    });

    // Attach click event listeners to dropdown items
    $('.dropdown-item').click(function() {
        // 'this' refers to the clicked .dropdown-item element
        var text = $(this).text();  // Using jQuery to get the text
        updateButtonText(text);
    });


    // Handling User Input Submission
    document.getElementById('submitQuery').addEventListener('click', function() {
        var userInput = document.getElementById('userInput').value;
        // Clear the input field after getting the value

        // Fetch request to Flask backend
        fetch('http://localhost:9000/process_user_input', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'user_input': userInput }),
        })
        .then(response => response.json())
        .then(data => {
            //debug
            console.log("Received data:", data); // Log the received data

            // Ensure that 'response' key exists in the data
            if(data.response) {
                // Display the response in your outputSection
                document.getElementById('outputSection').innerHTML = data.response;
            } else {
                // Handle the case where 'response' key doesn't exist
                console.error('Response key not found in data');
                document.getElementById('outputSection').innerHTML = 'An error occurred.' + error.message;
            }
        })
        .catch((error) => {
            // Handle any errors that occurred during fetch
            console.error('Error:', error);
            document.getElementById('outputSection').innerHTML = 'Error fetching data: ' + error.message;
        });
    });

});
