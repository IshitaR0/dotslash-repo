// FIRST----------------------------------------
// async function loadProducts() {
//     try {
//         const response = await fetch('sample.csv');
//         if (!response.ok) throw new Error('Failed to fetch the CSV file.');

//         const data = await response.text();

//         const rows = data.trim().split('\n');
//         const headers = rows[0].split(',');

//         const products = rows.slice(1).map(row => {
//             const columns = row.split(',');
//             const product = {};
//             headers.forEach((header, index) => {
//                 product[header.trim()] = columns[index]?.trim();
//             });
//             return product;
//         });

//         displayProducts(products); // Show products as flexbox cards
//     } catch (error) {
//         console.error('Error loading products:', error);
//     }
// }

// // Function to display products using flexboxes
// function displayProducts(products) {
//     const productContainer = document.getElementById('product-container');

//     // Clear any existing content in the container
//     productContainer.innerHTML = '';

//     products.forEach(product => {
//         // Create product card
//         const productCard = document.createElement('div');
//         productCard.classList.add('product-card');

//         productCard.innerHTML = `
//             <h2>${product.Name}</h2>
//             <p class="price">Rs. ${product.Price}</p>
//             <button class="add-to-cart">Add to Cart</button>
//         `;

//         productContainer.appendChild(productCard);
//     });
// }

// // Load products on page load
// loadProducts();

// SECOND----------------------------------------
async function handleSearchQuery(query) {
    try {
        // Send the query to the backend via a POST request
        const response = await fetch('http://localhost:5000/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json' // Specify JSON data type
            },
            body: JSON.stringify({ query: query }) // Send query as JSON
        });

        if (!response.ok) throw new Error('Failed to fetch search results.');

        const data = await response.json(); // Parse JSON response from the backend

        // Display the products based on the response
        displayProducts(data);
    } catch (error) {
        console.error('Error handling search query:', error);
    }
}

// Function to display products using flexboxes (updated to handle API response)
function displayProducts(products) {
    const productContainer = document.getElementById('product-container');
    productContainer.innerHTML = ''; // Clear the container first

    products.forEach(product => {
        // Create product card dynamically based on the received data
        const productCard = document.createElement('div');
        productCard.classList.add('product-card');

        productCard.innerHTML = `
            <h2>${product.Name}</h2>
            <p class="price">Rs. ${product.Price}</p>
            <button class="add-to-cart">Add to Cart</button>
        `;

        productContainer.appendChild(productCard);
    });
}

// Event listener for search input
document.getElementById('search-input').addEventListener('input', (event) => {
    const query = event.target.value;
    if (query.trim()) {
        handleSearchQuery(query); // Call the function to search when user types
    }
});