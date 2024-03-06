// JavaScript function to update button text and chart
function updateChartAndButtonText(text, symbol) {
    var button = document.getElementById('dropdownMenuButton1');
    button.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';

    // Update TradingView widget with the selected symbol
    new TradingView.widget({
        "container_id": "tradingview-widget",
        "autosize": false,
        "width": "100%",
        "height": "96%",
        "symbol": symbol, // Use the symbol parameter to dynamically set the symbol
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "details": true,
        "hotlist": true,
        "calendar": true,
        "studies": [
            "Volume@tv-basicstudies"
        ],
        "show_popup_button": true,
        "popup_width": "1000",
        "popup_height": "650"
    });
}

function updateDropdown(stocks) {
    const dropdown = document.getElementById('dropdownMenuButton1');
    dropdown.innerHTML = ''; // Clear existing options
    stocks.forEach(stock => {
        const option = document.createElement('a');
        option.classList.add('dropdown-item');
        option.href = '#';
        option.text = stock;
        option.setAttribute('data-symbol', stock); // Set the symbol as a data attribute
        dropdown.appendChild(option);
    });
}

function getPredictions(ticker) {
    $.ajax({
        url: '/get_predictions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ 'ticker': ticker }),
        success: function(predictions) {
            // Assuming predictions is an array of prediction values
            $('#predictionSection').empty(); // Clear previous predictions
            predictions.forEach(function(prediction) {
                $('#predictionSection').append(`<p>Prediction: ${prediction}</p>`);
            });
        },
        error: function(error) {
            console.log(error);
        }
    });
}

function displayPredictions(predictions) {
    // Assume predictions is an array of objects with 'symbol', 'percentChange', and 'volume'
    // First, sort the predictions based on percentChange
    predictions.sort((a, b) => b.percentChange - a.percentChange);

    // Get the dropdown element
    const dropdownMenu = document.getElementById('dropdownMenuButton1').nextElementSibling;

    // Clear existing dropdown items
    dropdownMenu.innerHTML = '';

    // Iterate over sorted predictions to create new dropdown items
    predictions.forEach((prediction, index) => {
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.classList.add('dropdown-item');
        link.href = '#';
        link.dataset.symbol = prediction.symbol;

        // Determine recommendation based on rank and percentChange
        let recommendation = '';
        if (index === 0) {
            recommendation = prediction.percentChange >= 0 ? 'Strong Buy' : 'Strong Sell';
        } else {
            recommendation = prediction.percentChange >= 0 ? 'Buy' : 'Sell';
        }

        // Set the display text with symbol, rank, and recommendation
        link.innerText = `${index + 1}. ${prediction.symbol} - ${recommendation}`;

        // Append the new dropdown item to the dropdown menu
        listItem.appendChild(link);
        dropdownMenu.appendChild(listItem);

        // Optionally, add an event listener to the link for further interaction
        link.addEventListener('click', function() {
            // Implement what should happen when a dropdown item is clicked
            console.log(`Selected ${prediction.symbol} with recommendation: ${recommendation}`);
        });
    });
}



$(document).ready(function() {
    // Handle dropdown item click
    $(document).on('click', '.dropdown-item', function() {
        var symbol = $(this).data('symbol');
        var text = $(this).text();
        updateChartAndButtonText(text, symbol);
    });

    // Initialize with a default symbol, if needed
    updateChartAndButtonText("Choose Instrument", "NASDAQ:AAPL");
});

$('#GetPredictionsBtn').click(function() {
    if ($('#lstmCheckbox').is(':checked')) {
        let selectedStocks = $('.dropdown-menu .dropdown-item').map(function() {
            return $(this).data('symbol');
        }).get();

        // Send the selected stocks to the backend for predictions
        $.ajax({
            url: '/get_predictions',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ selectedStocks: selectedStocks }),
            success: function(predictions) {
                // Clear existing recommendations
                $('.dropdown-menu .dropdown-item').each(function() {
                    $(this).text($(this).data('symbol'));
                });

                // Update dropdown items with predictions
                $.each(predictions, function(ticker, prediction) {
                    $('.dropdown-menu .dropdown-item').each(function() {
                        if ($(this).data('symbol') === ticker) {
                            $(this).text(ticker + ' - ' + prediction);
                        }
                    });
                });
            },
            error: function(error) {
                console.log('Error getting predictions:', error);
            }
        });
    }
});

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
        // Attach event listeners to newly added dropdown items
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', function() {
                const symbol = this.getAttribute('data-symbol');
                const text = this.text;
                updateChartAndButtonText(text, symbol);
            });
        });
    });

    // Attach click event listeners to dropdown items
    $(document).on('click', '.dropdown-item', function() {
        var text = $(this).text(); // Get the text of the clicked item
        var symbol = $(this).attr('data-symbol'); // Get the symbol from the data-symbol attribute

        updateChartAndButtonText(text, symbol);
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

    document.getElementById('ScanMarketBtn').addEventListener('click', function() {
        const loadingBar = document.getElementById('loadingBar');
        const progressBar = loadingBar.querySelector('.progress-bar');
        const progressPercentage = loadingBar.querySelector('#progressPercentage');
        let width = 0; // Initial width of the progress bar

        // Show the loading bar
        document.getElementById('loadingBar').style.display = '';

        // Simulate loading progress
        const interval = setInterval(function() {
            if (width >= 100) {
                clearInterval(interval);
            } else {
                width++;
                progressBar.style.width = width + '%';
                progressPercentage.textContent = width + '%';
            }
        }, 250); // Update the progress every 250 milliseconds

        fetch('/scan-market')
            .then(response => response.json())
            .then(data => {
                const dropdownMenu = document.getElementById('dropdownMenuButton1').nextElementSibling;
                // Clear existing dropdown items
                dropdownMenu.innerHTML = '';
                // Add new dropdown items
                data.forEach(stock => {
                    const dropdownItem = document.createElement('a');
                    dropdownItem.classList.add('dropdown-item');
                    dropdownItem.href = "#";
                    dropdownItem.setAttribute('data-symbol', stock);
                    dropdownItem.textContent = stock;
                    dropdownItem.addEventListener('click', function() {
                        updateChartAndButtonText(this.textContent, this.getAttribute('data-symbol'));
                    });
                    dropdownMenu.appendChild(dropdownItem);
                });

                // Hide the loading bar after data is loaded
                document.getElementById('loadingBar').style.display = 'none';
            }).catch(error => {
            console.error('Error fetching market data:', error);
            // Hide the loading bar in case of error
            document.getElementById('loadingBar').style.display = 'none';
        });
    });


});
