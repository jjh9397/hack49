/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',              // Main entry point
    './src/**/*.html',           // All HTML files in the src folder
  ],
  theme: {
    extend: {
      backgroundImage: {
        'main-background': "url('/Images/Background 1.jpeg')"
      }
    },
    screens: {
      'xs': '320px',   // Extra small devices
      'sm': '425px',   // Small devices (phones)
      'md': '768px',   // Medium devices (tablets)
      'lg': '1024px',  // Large devices (desktops)
      'xl': '1350px',  // Extra large devices (large desktops)
      '2xl': '1536px', // 2x extra large devices (larger desktops)
      '4xl': '1920px', // 4x extra large devices
      '5xl': '2136px', // 5x extra large devices
      '6xl': '2560px', // 6x extra large devices
    },
    fontFamily: {
      'inter-tight': ['Inter', 'sans-serif']
    }
  },
  plugins: [],
}