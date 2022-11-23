import numpy as np
from matplotlib import pyplot as plt
from CifExtractor import CifExtractor
from ElectroStaticsCalculations import *

# TODO: 
# 0.5 create simulation of ti atoms in a circle, should be zero
# 1. test code with cif file that has inversion symmetry P63mmc or cube (BTO)
# 1.5 check center of cube and move point to other location to check if e field changes
# 2. function that iterates thru every coord and finds the coord w/ the lowest e field
# divide dipole moment by volume to make dimensionless
# can also use BC to expand the box, first expand in x, then in y, then in z, total 6 steps
def main():
    np.set_printoptions(precision=4)

    cif = CifExtractor('CIF_files/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file

    hex_coords_by_element = cif.get_hex_coords_by_element() # 5 x 3
    cell_lengths = cif.get_cell_lengths() # 3 x 1
    hex_transformation_mat = cif.get_hex_transformation_matrix() # 12 x 3

    unit_cell = get_unit_cell_coord(hex_transformation_mat, hex_coords_by_element, cell_lengths) # 5 x 12 x 3
    print(unit_cell.shape, "Unit cell should be 5x12x3")
    directions = [0, 1, -1, 2]
    translated_unit_cells = get_translated_cells(unit_cell, cell_lengths, directions)
    print(translated_unit_cells.shape, "Translated unit cells should be (63,5,12,3) due to 63 translations")

    e_field_box = np.append([unit_cell], translated_unit_cells, axis=0)
    print(e_field_box.shape, "E field box (translations + OG UC) shape should be (64,5,12,3)")
    print("Example of Ba_1(untranslated)\n", e_field_box[0][0])
    last_translation = get_translation_permutations(cell_lengths, directions)[-1]
    print(f"Example of Ba_1(translated({last_translation}))\n", e_field_box[-1][0])
    cart_e_field_box = get_cart_e_field_box(e_field_box)
    print("Cartesian Example of Ba_1(untranslated)\n", cart_e_field_box[0][0])
    print(f"Cartesian Example of Ba_1(translated({last_translation}))\n", cart_e_field_box[-1][0])

    # Ba = 2+, Ti = 4+, S = 2-
    charges = np.array([2*1.6e-19, 4*1.6e-19, 4*1.6e-19, -2*1.6e-19, -2*1.6e-19])
    rand_coord = get_rand_coord(cart_e_field_box)
    print("E field [V/m] = ", f"{calc_e_field(cart_e_field_box, rand_coord, charges):.4e}")
    print('Electric dipole moment [C m]', f"{calc_dipole_moment(cart_e_field_box, rand_coord, charges):.4e}")

#gets random coordinate values
def get_rand_coord(cart_e_field_box):
    rand_index = get_rand_index(cart_e_field_box)
    return cart_e_field_box[rand_index[0], rand_index[1], rand_index[2]] #cart_e_field_box[6, 1, 2]

#gets index of random coordinate
def get_rand_index(cart_e_field_box):
    rand_index = np.random.randint(0, cart_e_field_box.shape)
    return rand_index    

if __name__ == '__main__':
    main()