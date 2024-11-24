async function loadProducts() {
    try {
        const response = await fetch('sample.csv');
        if (!response.ok) throw new Error('Failed to fetch the CSV file.');

        const data = await response.text();

        const rows = data.trim().split('\n');
        const headers = rows[0].split(',');

        const products = rows.slice(1).map(row => {
            const columns = row.split(',');
            const product = {};
            headers.forEach((header, index) => {
                product[header.trim()] = columns[index]?.trim();
            });
            return product;
        });

        displayProducts(products); // Show products as flexbox cards
    } catch (error) {
        console.error('Error loading products:', error);
    }
}

function displayProducts(products) {
    const productContainer = document.getElementById('product-container');

    // Clear any existing content in the container
    productContainer.innerHTML = '';

    products.forEach(product => {
        const productCard = document.createElement('div');
        productCard.classList.add('product-card');

        productCard.innerHTML = `
            <h3>${product.Name}</h3>
            <p class="price">Rs. ${product.Price}</p>
        `;

        productContainer.appendChild(productCard);
    });
}

// Load products on page load
loadProducts();
