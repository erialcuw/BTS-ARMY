import sys
from gemmi import cif

greeted = set()
try:
    doc = cif.read_file('/Users/clairewu/Downloads/gemmi/BTS_Plate_300K_P63cm.cif')  # copy all the data from mmCIF file
    block = doc.sole_block()  # CIF has exactly one block
    for element in block.find_value('_cell_length_a'): # block.find_loop("_atom_site_type_symbol")
        if element not in greeted:
            print("Hello " + element)
            greeted.add(element)
    print('done')
except Exception as e:
    print("Oops.", e)
    sys.exit(1)