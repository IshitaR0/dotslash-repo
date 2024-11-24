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
// async function handleSearchQuery(query) {
//     try {
//         // Send the query to the backend via a POST request
//         const response = await fetch('http://localhost:5000/search', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json' // Specify JSON data type
//             },
//             body: JSON.stringify({ query: query }) // Send query as JSON
//         });

//         if (!response.ok) throw new Error('Failed to fetch search results.');

//         const data = await response.json(); // Parse JSON response from the backend

//         // Display the products based on the response
//         displayProducts(data);
//     } catch (error) {
//         console.error('Error handling search query:', error);
//     }
// }

// // Function to display products using flexboxes (updated to handle API response)
// function displayProducts(products) {
//     const productContainer = document.getElementById('product-container');
//     productContainer.innerHTML = ''; // Clear the container first

//     products.forEach(product => {
//         // Create product card dynamically based on the received data
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

// // Event listener for search input
// document.getElementById('search-input').addEventListener('input', (event) => {
//     const query = event.target.value;
//     if (query.trim()) {
//         handleSearchQuery(query); // Call the function to search when user types
//     }
// });

// THIRD-------------------------------------------------------------------
// document
//   .getElementById("searchForm")
//   .addEventListener("submit", async function (event) {
//     event.preventDefault(); // Prevent form from refreshing the page
//     const query = document.getElementById("queryInput").value;

//     // Send query to backend
//     const response = await fetch("/search", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ query }),
//     });

//     const results = await response.json(); // Parse the JSON response
//     displayResults(results); // Function to show results on the frontend
//   });

// // Function to display search results
// function displayResults(results) {
//   const resultsDiv = document.getElementById("results");
//   resultsDiv.innerHTML = results
//     .map(
//       (item) => `
//             <div class="result-card">
//                 <h3>${item.name}</h3>
//                 <p>${item.description}</p>
//                 <p>Price: ${item.price}</p>
//             </div>
//         `
//     )
//     .join("");
// }
// document
//   .getElementById("searchForm")
//   .addEventListener("submit", async function (event) {
//     event.preventDefault(); // Prevent page refresh
//     const query = document.getElementById("queryInput").value;

//     try {
//       // Send the query to the backend
//       const response = await fetch("http://127.0.0.1:5000/search", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ query }), // Send the query in JSON format
//       });

//       // Check if the response status is not OK
//       if (!response.ok) {
//         throw new Error("Failed to fetch results");
//       }

//       // Parse the response as JSON
//       const results = await response.json();
//       displayResults(results); // Display the results in the UI
//     } catch (error) {
//       console.error(error);
//       document.getElementById("results").innerHTML =
//         '<p style="color: red;">An error occurred while fetching results.</p>';
//     }
//   });

// function displayResults(results) {
//   const resultsDiv = document.getElementById("results");

//   // Handle empty results
//   if (results.length === 0) {
//     resultsDiv.innerHTML = "<p>No products found.</p>";
//     return;
//   }

//   // Create HTML for the results
//   resultsDiv.innerHTML = results
//     .map(
//       (item) => `
//         <div class="result-card">
//           <h3>${item.name}</h3>
//           <p>${item.description}</p>
//           <p>Price: ${item.price}</p>
//         </div>
//       `
//     )
//     .join("");
// }

// FOURTH---------------------------
document
  .getElementById("searchForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent page refresh
    const query = document.getElementById("queryInput").value.trim();

    if (!query) {
      document.getElementById("error-message").textContent =
        "Please enter a valid query.";
      return;
    }

    document.getElementById("error-message").textContent = ""; // Clear any previous errors

    try {
      // Send the query to the backend
      const response = await fetch("http://127.0.0.1:5001/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query }),
      });

      // Check if the response status is not OK
      if (!response.ok) {
        throw new Error("Failed to fetch results");
      }

      // Parse the response as JSON
      const results = await response.json();
      displayResults(results); // Display the results in the UI
    } catch (error) {
      console.error(error);
      document.getElementById("error-message").textContent =
        "An error occurred while fetching results.";
    }
  });

function displayResults(results) {
  const resultsDiv = document.getElementById("results");

  // Handle empty results
  if (results.length === 0) {
    resultsDiv.innerHTML = "<p>No products found.</p>";
    return;
  }

  // Create HTML for the results
  resultsDiv.innerHTML = results
    .map(
      (item) => `
        <div class="result-card">
          <h3>${item.Name}</h3>
          <p>${item.Description}</p>
          <p>Price: Rs. ${item.Price}</p>
        </div>
      `
    )
    .join("");
}
