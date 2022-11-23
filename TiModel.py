from sympy import Matrix

# This is a dummy CifExtractor for testing with just Ti
class TiModel:
    def __init__(self):
        pass
    
    def get_hex_coords_by_element(self):
        return [[0.6667, 0.3333, 0.0668]]
    
    def get_cell_lengths(self):
        return [1.1671e-09, 1.1671e-09, 5.833e-10]

    def get_hex_transformation_matrix(self):
        return Matrix([["x","y","z"]])

def main():
    np.set_printoptions(precision=4)

    cif = TiModel()  # Use the TiModel cif data

    hex_coords_by_element = cif.get_hex_coords_by_element() # 1 x 3
    cell_lengths = cif.get_cell_lengths() # 3 x 1
    hex_transformation_mat = cif.get_hex_transformation_matrix() # 1 x 3

    unit_cell = get_unit_cell_coord(hex_transformation_mat, hex_coords_by_element, cell_lengths) # 1 x 12 x 3
    print(unit_cell.shape, "Unit cell should be 1x1x3")
    directions = [0, 1, -1]
    translated_unit_cells = get_translated_cells(unit_cell, cell_lengths, directions)
    print(translated_unit_cells.shape, "Translated unit cells should be (26,1,1,3) due to 26 translations")

    e_field_box = np.append([unit_cell], translated_unit_cells, axis=0)
    print(e_field_box.shape, "E field box (translations + OG UC) shape should be (27,1,1,3)")
    print("Example of Ti_1(untranslated)\n", e_field_box[0][0])
    last_translation = get_translation_permutations(cell_lengths, directions)[-1]
    print(f"Example of Ti_1(translated({last_translation}))\n", e_field_box[-1][0])
    cart_e_field_box = get_cart_e_field_box(e_field_box)
    print("Cartesian Example of Ti_1(untranslated)\n", cart_e_field_box[0][0])
    print(f"Cartesian Example of Ti_1(translated({last_translation}))\n", cart_e_field_box[-1][0])

    # Ti = 4+
    charges = np.array([4*1.6e-19])
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
    import numpy as np
    from ElectroStaticsCalculations import *
    main()