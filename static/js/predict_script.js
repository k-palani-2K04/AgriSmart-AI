document.addEventListener('DOMContentLoaded', function() {
    const el = document.querySelector('#State_Name');
    if (el && typeof Choices !== 'undefined') {
        try {
            new Choices('#State_Name', {
                searchEnabled: true,       // Enables search functionality
                itemSelectText: '',        // Removes "Press to select" text
                placeholderValue: 'Enter state name', // Adds placeholder text
            });
        } catch (e) {
            console.log('Choices JS init skipped:', e);
        }
    }
});