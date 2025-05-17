import { bg_colors } from './colors.js';

// JavaScript to generate the grid will go here
document.addEventListener('DOMContentLoaded', function() {
  const svgNS = "http://www.w3.org/2000/svg";
  const gridContainer = document.getElementById('gridContainer');
  const svg = document.getElementById('hexagonGridSVG');

  const numRows = bg_colors.length;
  const numCols = bg_colors[0].length;

  // Define the dimensions of the hexagon based on the symbol's viewBox
  // These are the dimensions of the hexagon's bounding box
  const visualHexWidth = 10.0; // From symbol viewBox width
  const visualHexHeight = 8.66;  // From symbol viewBox height


  // Calculate spacing for pointy-top hexagons
  // Horizontal distance between the start of one hex column and the next
  // const colSpacing = hexWidth * 2;
  // Vertical distance from the top of one row to the top of the next
  // For pointy-top, rows are packed tighter, so it's 3/4 of the height
  // const rowSpacing = hexHeight / 2.0;

  // Corrected spacing for FLAT-TOPPED hexagons:
  const horizontalGridStep = visualHexWidth * 0.75; // Correct: X distance between column centers/starts
  const verticalGridStep = visualHexHeight;       // Correct: Y distance between row centers/starts in the SAME column
  const yOffsetForStaggeredColumn = visualHexHeight / 2; // Correct: Vertical offset for odd/even columns

  // Example terrain data (0: water, 1: land, 2: mountain)
  // This should be a 20x10 array (numRows x numCols)
  const terrainData = [];
  const terrainTypes = ['water', 'land', 'mountain'];

  let maxX = 0;
  let maxY = 0;

  for (let row = 0; row < numRows; row++) {
    for (let col = 0; col < numCols; col++) {
      let xPos = col * horizontalGridStep;
      let yPos = row * verticalGridStep; // Base Y position for the row

      // Stagger odd columns vertically
      if (col % 2 !== 0) {
        yPos += yOffsetForStaggeredColumn;
      }

      const useElement = document.createElementNS(svgNS, 'use');
      useElement.setAttributeNS('http://www.w3.org/1999/xlink', 'href', '#hexagonSymbol');
      useElement.setAttribute('x', xPos);
      useElement.setAttribute('y', yPos);
      useElement.setAttribute('width', visualHexWidth);   // Explicitly set width
      useElement.setAttribute('height', visualHexHeight); // Explicitly set height
      useElement.setAttribute('fill', bg_colors[row][col]); // Explicitly set height

      // Get terrain type and apply class
      useElement.classList.add('hexagon');

      // Add an ID for potential individual manipulation or event handling
      useElement.id = `hex_${col + 2}_${row + 1}`;

      gridContainer.appendChild(useElement);

      // Track maximum dimensions to potentially resize SVG
      if (xPos + visualHexWidth > maxX) maxX = xPos + visualHexWidth;
      if (yPos + visualHexHeight > maxY) maxY = yPos + visualHexHeight;
    }
  }

  // Optionally, adjust SVG canvas size based on content
  // Add some padding
  const padding = 2;
  svg.setAttribute('width', maxX + padding);
  svg.setAttribute('height', maxY + padding);

  // Example of adding an event listener to a hexagon
  gridContainer.addEventListener('click', function(event) {
    const target = event.target.closest('.hexagon'); // or event.target if <use> is directly targeted
    if (target) {
      console.log('Clicked hexagon ID:', target.id, 'Classes:', target.className.baseVal);
      alert(`Clicked hexagon ID: ${target.id}`);
      // You could change terrain type or display info here
    }
  });
});

