from qgis.core import QgsRasterLayer, QgsRasterCalculator, QgsRasterCalculatorEntry
import os

def apply_landmask(input_rasters, landmask_path, output_dir):
    """
    Apply a landmask operation on multiple rasters. The operation will multiply each raster's 
    band 1 with the landmask where the landmask value is 1.

    :param input_rasters: List of input raster file paths to be masked
    :param landmask_path: Path to the landmask raster
    :param output_dir: Directory to save the output masked raster files
    """
    # Check if the output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the landmask raster
    landmask = QgsRasterLayer(landmask_path, 'Landmask')

    if not landmask.isValid():
        print(f"Landmask raster is invalid: {landmask_path}")
        return

    # Loop through each input raster and apply the landmask
    for raster_path in input_rasters:
        # Load the input raster
        input_raster = QgsRasterLayer(raster_path, 'Input Raster')

        # Check if the layer is valid
        if not input_raster.isValid():
            print(f"Input raster is invalid: {raster_path}")
            continue

        # Create RasterCalculator entries for the input raster and landmask
        entry_input = QgsRasterCalculatorEntry()
        entry_input.ref = f'"{input_raster.name()}@1"'
        entry_input.raster = input_raster
        entry_input.bandNumber = 1

        entry_landmask = QgsRasterCalculatorEntry()
        entry_landmask.ref = f'"{landmask.name()}@1"'
        entry_landmask.raster = landmask
        entry_landmask.bandNumber = 1

        # Prepare the formula for the landmask operation: input_raster * (landmask == 1)
        formula = f'"{input_raster.name()}@1" * ("{landmask.name()}@1" = 1)'

        # Create output file name based on input raster name
        output_filename = f"masked_{os.path.basename(raster_path)}"
        output_path = os.path.join(output_dir, output_filename)

        # Create the RasterCalculator and run the calculation
        calc = QgsRasterCalculator(formula, output_path, 'GTiff', input_raster.extent(), input_raster.width(), input_raster.height(), [entry_input, entry_landmask])
        calc.processCalculation()

        print(f"Landmask applied to {raster_path} and saved to: {output_path}")

# Example usage: Apply landmask operation to multiple input rasters
input_rasters = [
    '/path/to/your/raster1.tif',
    '/path/to/your/raster2.tif',
    '/path/to/your/raster3.tif'
]

landmask_path = '/path/to/your/landmask.tif'
output_dir = '/path/to/save/masked_rasters'

apply_landmask(input_rasters, landmask_path, output_dir)
