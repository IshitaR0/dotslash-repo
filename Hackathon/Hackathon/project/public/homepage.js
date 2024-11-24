document.addEventListener('DOMContentLoaded', () => {
    const nextButton = document.getElementById('next-button');
    const clearButton = document.getElementById('clear-button');
    const warningText = document.getElementById('warning');

    nextButton.addEventListener('click', () => {
        const selectedOption = document.querySelector('input[name="option"]:checked');
        if (!selectedOption) {
            warningText.textContent = 'You are required to choose an option before proceeding.';
            return;
        }
        warningText.textContent = ''; // Clear warning text
        window.location.href = 'maincode.html'; // Navigate to the product page
    });

    clearButton.addEventListener('click', () => {
        const selectedOption = document.querySelector('input[name="option"]:checked');
        if (selectedOption) selectedOption.checked = false;
        warningText.textContent = ''; // Clear warning text
    });
});
