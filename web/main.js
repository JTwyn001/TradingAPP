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
    // Trading_ai-html
    // JavaScript function to update button text while preserving the caret icon
    window.updateButtonText = function(text) {
        var button = document.getElementById('dropdownMenuButton1');
        button.innerHTML = text + ' <i class="bi bi-caret-down-fill"></i>';
    };

});