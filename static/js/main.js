// JavaScript function to update button text and chart
function updateChartAndButtonText(text, symbol) {
    var suffixesToRemove = ['.NYSE', '.NAS', '.LSE', '.ASE', '.TSX', '.ETR']; // Add more as needed
    var button = document.getElementById('dropdownMenuStockButton');
    button.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';
    var button2 = document.getElementById('dropdownMenuForexButton');
    button2.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';

    // Remove suffixes from symbol
    var cleanedSymbol = symbol;
    suffixesToRemove.forEach(function(suffix) {
        cleanedSymbol = cleanedSymbol.replace(suffix, '');
    });

    // Update TradingView widget with the selected symbol
    new TradingView.widget({
        "container_id": "tradingview-widget",
        "autosize": false,
        "width": "100%",
        "height": "96%",
        "symbol": cleanedSymbol, // Use the symbol parameter to dynamically set the symbol
        "interval": "1",
        "timezone": "Etc/UTC",
        "theme": "dark",
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


$(document).ready(function() {
    // Handle dropdown item click
    $(document).on('click', '.dropdown-item .dropdown-item-stock', function() {
        var symbol = $(this).data('symbol');
        var text = $(this).text();
        updateChartAndButtonText(text, symbol);
    });

    $(document).on('click', '.dropdown-item-forex', function() {
        var symbol = $(this).data('symbol');
        var text = $(this).text();
        updateChartAndButtonText(text, symbol);
    });

    // Initialize with a default symbol, if needed
    updateChartAndButtonText("Choose Stock Instrument", "BTCUSD");
});

$(document).ready(function() {
    // Handle dropdown item click
    $(document).on('click', '.dropdown-item-forex', function() {
        var symbol = $(this).data('symbol');
        var text = $(this).text();
        updateChartAndButtonText(text, symbol);
    });

    // Initialize with a default symbol, if needed
    updateChartAndButtonText("Choose Forex Instrument", "BTCUSD");
});

function calculateVolume(ticker, price, dollarAllocation) {
    console.log(`Calculating volume for ${ticker} with price: ${price}, dollarAllocation: ${dollarAllocation}`);
    if (price <= 0 || isNaN(price)) {
        console.error('Invalid price detected:', price);
        return 0;  // Default to 0 or an appropriate error handling
    }

    // Determine volume based on ticker type
    let volume = 0;
    if (ticker.endsWith('.NAS') || ticker.endsWith('.NYSE')) {
        // For stocks, where 0.1 volume = price/10
        volume = (dollarAllocation / (price / 10)) * 0.1;
    } else if (['SEKJPY', 'EURNOK', 'EURUSD'].includes(ticker)) {
        // For forex pairs, where 0.01 minimum and = price x 1000
        volume = (dollarAllocation * 1000) / price;
    }

    // Adjust volume to meet minimums
    let minimumVolume = ticker.endsWith('.NAS') || ticker.endsWith('.NYSE') ? 0.1 : 0.01;
    volume = Math.max(volume, minimumVolume);
    volume = volume.toFixed(2); // Adjust decimal places as necessary
    console.log(`Adjusted volume calculated: ${volume}`);
    return volume;
}

let tradingData = {};  // Global object to store trading data
$('#ExecuteMLBtn').click(function() {
    $.ajax({
        url: '/execute_ml_predictions',
        type: 'GET',
        success: function(response) {
            let totalCapital = parseFloat($('#capitalSlider').val());
            let maxAllocationPerTicker = totalCapital * 0.4; // 40% cap
            let weights = {};
            let allocations = {};
            let excessCapital = 0;
            let totalWeightForUncapped = 0;
            let sumOfWeights = 0;

            // Calculate weights and initial allocations
            Object.entries(response).forEach(([ticker, predictions]) => {
                let pctChange = ((predictions.avg - predictions.last_close) / predictions.last_close) * 100;
                weights[ticker] = Math.exp(Math.abs(pctChange)); // Exponential of the absolute percentage change
                sumOfWeights += weights[ticker];
            });

            // Normalize and apply caps
            Object.entries(weights).forEach(([ticker, weight]) => {
                let allocation = (weight / sumOfWeights) * totalCapital; // Normalize based on total weight
                if (allocation > maxAllocationPerTicker) {
                    excessCapital += allocation - maxAllocationPerTicker; // Calculate excess
                    allocations[ticker] = maxAllocationPerTicker; // Cap the allocation
                } else {
                    allocations[ticker] = allocation;
                    totalWeightForUncapped += weight; // Sum weights for uncapped allocations
                }
            });

            // Redistribute excess capital proportionally to uncapped tickers
            if (excessCapital > 0) {
                Object.keys(allocations).forEach(ticker => {
                    if (allocations[ticker] < maxAllocationPerTicker) {
                        // Only redistribute to tickers that were not initially capped
                        let additionalAllocation = (weights[ticker] / totalWeightForUncapped) * excessCapital;
                        allocations[ticker] += additionalAllocation; // Add proportion of excess
                    }
                });
            }

            // Calculate volumes and update UI
            Object.entries(response).forEach(([ticker, predictions]) => {
                let dollarAmount = allocations[ticker];
                let price = parseFloat(predictions.current_price);
                let volume = calculateVolume(ticker, price, dollarAmount);
                let cleanTicker = ticker.replace(/\.\w+$/, ''); // Removes any exchange suffix like .NAS or .NYSE
                let pctChange = ((predictions.avg - predictions.last_close) / predictions.last_close) * 100;

                updateUI(cleanTicker, predictions, dollarAmount, volume, pctChange, totalCapital);
            });

            $('#TradeStockBtn').prop('disabled', false); // Enable trade button
        },
        error: function(xhr, status, error) {
            console.error('Error executing ML predictions:', error);
        }
    });
});

function updateUI(ticker, predictions, allocation, volume, pctChange, totalCapital) {
    $(`#lstmPrediction-${ticker}`).text(predictions.lstm ? predictions.lstm.toFixed(3) : 'N/A');
    $(`#gbmPrediction-${ticker}`).text(predictions.gbm ? predictions.gbm.toFixed(3) : 'N/A');
    $(`#avgPrediction-${ticker}`).text(predictions.avg ? predictions.avg.toFixed(3) : 'N/A');
    $(`#lastClose-${ticker}`).text(predictions.last_close ? predictions.last_close.toFixed(2) : 'N/A');
    $(`#pctChange-${ticker}`).text(pctChange.toFixed(2) + '%');
    $(`#allocation-${ticker}`).text((allocation / totalCapital * 100).toFixed(2) + '%');
    $(`#dollarAllocation-${ticker}`).text('$' + allocation.toFixed(2));
    $(`#volume-${ticker}`).text(volume);
}


$('#GetPredictionsBtn').click(function() {
    var selectedStocks = [];
    if ($('#lstmCheckbox').is(':checked')) {
        selectedStocks = $('.dropdown-menu .dropdown-item').map(function() {
            return $(this).data('symbol');
        }).get();
    }

    // Send the selected stocks to the backend for predictions
    $.ajax({
        url: '/get_predictions',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({ selectedStocks: selectedStocks }),
        success: function(response) {
            console.log('Response from server:', response); // Add this line to log the response
            // Clear existing recommendations
            $('.dropdown-menu .dropdown-item').each(function() {
                $(this).text($(this).data('symbol')).removeClass('buy-prediction sell-prediction');
            });
            // Then update dropdown items with predictions
            for (var ticker in response) {
                var prediction = response[ticker];
                var $dropdownItem = $('a[data-symbol="' + ticker + '"]');
                // Update the text
                $dropdownItem.text(ticker + ' - ' + prediction);
                // Add color based on the prediction
                if (prediction === 'BUY') {
                    $dropdownItem.addClass('buy-prediction');
                } else if (prediction === 'SELL') {
                    $dropdownItem.addClass('sell-prediction');
                }
                // Find the corresponding dropdown item and update its text
                // $('a[data-symbol="' + ticker + '"]').text(ticker + ' -   ' + '     ' + prediction);
            }
            document.getElementById('predictionMessage').style.display = 'block';
        },
        error: function(error) {
            console.log('Error getting predictions:', error);
        }
    });
});

document.getElementById('AccountBtn').addEventListener('click', function() {
    fetch('/update_positions')
        .then(response => response.json())
        .then(data => {
            // Use the data to update the HTML elements for open positions
            // This part will depend on how your HTML is structured
            const positionsContainer = document.getElementById('positions');
            positionsContainer.innerHTML = ''; // Clear existing content
            data.forEach(pos => {
                const posElement = document.createElement('div');
                posElement.innerHTML = `${pos.symbol}: <span style="color: ${pos.profit >= 0 ? 'green' : 'red'}">${pos.profit.toFixed(2)}</span>`;
                positionsContainer.appendChild(posElement);
            });
        })
        .catch(error => console.log('Error:', error));
});

let predictionData = {};


// Function to execute trades
function executeTrades() {
    // Filter out tickers with non-numeric prediction values
    let validPredictionData = {};
    for (let [ticker, value] of Object.entries(predictionData)) {
        if (!isNaN(value)) {  // Check if the value is numeric
            validPredictionData[ticker] = value;
        } else {
            console.error(`Skipping trade for ${ticker} due to invalid prediction value: ${value}`);
        }
    }

    return fetch('/execute_lstm_trades', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ predictionValues: validPredictionData }),
    })
    .then(response => response.json())
    .then(data => console.log('Trade Execution Response:', data))
    .catch((error) => console.error('Error executing trades:', error));
}

function getSelectedStocks() {
    let selectedStocks = [];
    // Iterate over each dropdown item and extract the symbol
    $('.dropdown-menu .dropdown-item').each(function() {
        selectedStocks.push($(this).data('symbol'));
    });
    return selectedStocks;
}

function getSelectedForex() {
    let selectedForex = [];
    // Iterate over each dropdown item and extract the symbol
    $('.dropdown-menu .dropdown-item-forex').each(function() {
        selectedForex.push($(this).data('symbol'));
    });
    return selectedForex;
}


$('#TradeStockBtn').click(function() {
    console.log('Trade button clicked');
    let trades = Object.values(tradingData); // Convert stored data into an array for sending
    if (trades.length === 0) {
        console.error('No trade data to send');
        alert('No trade data to send. Ensure ML predictions have been executed and table is populated.');
        return;
    }

    console.log('Number of trades to send:', trades.length);
    console.log('Sending trades to server...', trades);

    $.ajax({
        url: '/trade_stocks',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(trades),
        success: function(response) {
            let errors = Object.values(response).filter(trade => trade.status === 'error');
            if (errors.length > 0) {
                alert("Some trades failed to execute: " + JSON.stringify(errors));
            } else {
                alert("Trades have been executed successfully!");
            }
            console.log('Trade execution response:', response);

        },
        error: function(xhr, status, error) {
            console.error('Error executing trades:', error);
            alert('Failed to execute trades: ' + xhr.responseText);
        }
    });
});


function updateAccountInfo() {
    fetch('/get_account_info')
        .then(response => response.json())
        .then(data => {
            if(data.error) {
                console.log('Error fetching account info:', data.error);
            } else {
                document.getElementById('balance').innerText = data.balance.toFixed(2);
                document.getElementById('equity').innerText = data.equity.toFixed(2);
            }
        })
        .catch(error => console.log('Error:', error));
}

document.addEventListener('DOMContentLoaded', function() {
    updateAccountInfo(); // Update on page load
});

document.getElementById('AccountBtn').addEventListener('click', function() {
    updateAccountInfo(); // Update on button click
});


setInterval(() => {
    fetch('/get_account_info')
        .then(response => response.json())
        .then(data => {
            // Update balance and equity
            document.getElementById('balance').textContent = `${data.balance.toFixed(2)} GBP`;
            document.getElementById('equity').textContent = `${data.equity.toFixed(2)} GBP`;
            document.getElementById('balance').style.color = 'green';
            document.getElementById('equity').style.color = 'green';

            // Update open positions
            const positionsContainer = document.getElementById('positions');
            positionsContainer.innerHTML = ''; // Clear current positions
            data.positions.forEach(pos => {
                const posElement = document.createElement('div');
                posElement.textContent = `${pos.symbol}: ${pos.profit.toFixed(2)} ${pos.currency}`;
                posElement.style.color = pos.profit >= 0 ? 'green' : 'red';
                positionsContainer.appendChild(posElement);
            });
        })
        .catch(error => console.error('Failed to fetch account info:', error));
}, 1000); // Update every 2 seconds


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

    document.addEventListener('DOMContentLoaded', function() {
        // Attach event listeners to newly added dropdown items
        document.querySelectorAll('.dropdown-item-forex').forEach(item => {
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

    $(document).on('click', '.dropdown-item-forex', function() {
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
        }, 200); // Update the progress every 125 milliseconds, so it matches the process on terminal

        fetch('/scan-market')
            .then(response => response.json())
            .then(data => {
                const dropdownMenu = document.getElementById('dropdownMenuStockButton').nextElementSibling;
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
                // Make the success message visible
                document.getElementById('scan-marketMessage').style.display = 'block';
            }).catch(error => {
            console.error('Error fetching stock market data:', error);
            // Hide the loading bar in case of error
            document.getElementById('loadingBar').style.display = 'none';
        });
    });

    document.getElementById('ScanForexBtn').addEventListener('click', function() {
        const forexloadingBar = document.getElementById('forexloadingBar');
        const progressBar = forexloadingBar.querySelector('.progress-bar');
        const forexprogressPercentage = forexloadingBar.querySelector('#forexprogressPercentage');
        let width = 0; // Initial width of the progress bar

        forexloadingBar.style.display = '';

        // Show the loading bar
        // document.getElementById('loadingBar').style.display = '';
        const interval = setInterval(function() {
            if (width >= 100) {
                clearInterval(interval);
            } else {
                width++;
                progressBar.style.width = width + '%';
                forexprogressPercentage.textContent = width + '%';
            }
        }, 200); // Update the progress every 125 milliseconds, so it matches the process on terminal
        fetch('/scan-forex-market')
            .then(response => response.json())
            .then(data => {

                const forexDropdownMenu = document.getElementById('dropdownMenuForexButton').nextElementSibling;
                // dropdownMenu.innerHTML = '';
                while (forexDropdownMenu.children.length > 1) {
                    forexDropdownMenu.removeChild(forexDropdownMenu.lastChild);
                }
                data.forEach(forex => {
                    const dropdownItem = document.createElement('a');
                    dropdownItem.classList.add('dropdown-item');
                    dropdownItem.href = "#";
                    dropdownItem.setAttribute('data-symbol', forex);
                    dropdownItem.textContent = forex.replace(/\.NAS|\.NYSE/, ''); // Removing the exchange part
                    dropdownItem.addEventListener('click', function() {
                        updateChartAndButtonText(this.textContent, this.getAttribute('data-symbol'));
                    });
                    // dropdownMenu.appendChild(dropdownItem);
                    const li = document.createElement('li');
                    li.appendChild(dropdownItem);
                    forexDropdownMenu.appendChild(li);
                });
                forexloadingBar.style.display = 'none';
                // Make the success message visible
                document.getElementById('scan-forexMarketMessage').style.display = 'block';
            }).catch(error => {
            console.error('Error fetching forex market data:', error);
            forexloadingBar.style.display = 'none';
        });
    });

});
