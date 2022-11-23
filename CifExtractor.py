import numpy as np
from gemmi import cif
from sympy import Matrix

class CifExtractor:
    def __init__(self, file_path):
        doc = cif.read_file(file_path)  # copy all the data from mmCIF file
        self.block = doc.sole_block()  # CIF has exactly one block
    
    # extracts 12 HEX symmetry operations from CIF file 
    def get_hex_transformation_matrix(self):
        hex_transformation_mat = CifExtractor.get_transformation_mat('_space_group_symop_operation_xyz', self.block)
        return hex_transformation_mat

    # extracts HEX atom site fract & cell length from CIF file
    # each row(5) represents a single element
    # each column (3) represents an xyz direction
    def get_hex_coords_by_element(self):
        hex_coords_by_element = np.array([
            CifExtractor.get_fract_vals('_atom_site_fract_x', self.block),
            CifExtractor.get_fract_vals('_atom_site_fract_y', self.block),
            CifExtractor.get_fract_vals('_atom_site_fract_z', self.block)
        ]).transpose()
        # [Ba_1 Ti_1 Ti_2 S_1 S_2]
        return hex_coords_by_element
    
    # extracts symmetry operations from CIF file into a matrix of strings
    def get_transformation_mat(symm_operations, block):
        transformation_expressions = [cif.as_string(i) for i in (block.find_loop(symm_operations))]
        transformation_expressions_per_axis = Matrix([ 
            [coord_expr.strip() for coord_expr in expr.split(',')] 
            for expr in transformation_expressions
        ])
        return transformation_expressions_per_axis

    # extracts fract site values
    def get_fract_vals(site_fract_name, block):
        return [CifExtractor.as_number(val) for val in block.find_loop(site_fract_name)]
        
    # extracts all cell lengths (a,b,c)
    def get_cell_lengths(self):
        cell_length_names = ['_cell_length_a', '_cell_length_b', '_cell_length_c']
        return [CifExtractor.get_cell_length(name, self.block) for name in cell_length_names]

    #extract single cell length given name
    def get_cell_length(cell_length_name, block):
        return CifExtractor.as_number(block.find_value(cell_length_name)) * 1e-10

    # converts strings to numbers 
    def as_number(num_str):
        return cif.as_number(CifExtractor.remove_stddev(num_str))

    # removes std. dev from CIF file values
    def remove_stddev(num_str):
        index_of_stddev = num_str.find('(')
        if index_of_stddev >= 0:
            return num_str[:index_of_stddev]
        return num_str #string