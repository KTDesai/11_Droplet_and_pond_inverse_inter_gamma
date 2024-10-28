def generateBoundaryFile(boundary_fname):   
    boundary_old = open(boundary_fname, 'r')
    boundary = open('boundaries.txt', 'w')

    old_lines = boundary_old.readlines()

    boundary.writelines(old_lines[:2])

    old_lines = old_lines[2:]

    for i in range(0, len(old_lines), 6):
        boundary_type = old_lines[i].strip()
        if(boundary_type == "defaultFaces"):
            boundary.write(f"empty\n")
            n_faces_string = old_lines[i+4].strip()
            n_faces = int(n_faces_string[:-1].split()[1])
            start_face_string = old_lines[i+5].strip()
            start_face = int(start_face_string[:-1].split()[1])
            boundary.write(str(f"{n_faces}\n"))
            boundary.write("(\n")
            face_list = list(map(str,list(range(start_face, start_face + n_faces))))
            face_list_string = " ".join(face_list)
            boundary.write(f"{face_list_string}\n)\n)")
            break
        else:
            boundary.write(f"{boundary_type}\n")
            n_faces_string = old_lines[i+3].strip()
            n_faces = int(n_faces_string[:-1].split()[1])
            start_face_string = old_lines[i+4].strip()
            start_face = int(start_face_string[:-1].split()[1])
            boundary.write(str(f"{n_faces}\n"))
            boundary.write("(\n")
            face_list = list(map(str,list(range(start_face, start_face + n_faces))))
            face_list_string = " ".join(face_list)
            boundary.write(f"{face_list_string}\n)\n")   

    boundary_old.close()
    boundary.close()

def generateCellFile(owner_fname, neighbour_fname):
    owner_file = open(owner_fname, 'r')
    neighbour_file = open(neighbour_fname, 'r')
    cell_file =  open("cells.txt", 'w')
    cell_dict = {}
    owner_file_lines = owner_file.readlines()[2:-1]
    neighbour_file_lines = neighbour_file.readlines()[2:-1]
    for i, line in enumerate(owner_file_lines):
        cell_number = int(line.strip())
        if cell_number in cell_dict:
            cell_dict[cell_number].append(i)
        else:
            cell_dict[cell_number] = [i]

    for i, line in enumerate(neighbour_file_lines):
        cell_number = int(line.strip())
        if cell_number in cell_dict:
            cell_dict[cell_number].append(i)
        else:
            cell_dict[cell_number] = [i]

    cell_file.write(f"{str(len(cell_dict))}\n")
    cell_file.write("(\n")
    for key, val in cell_dict.items():#
        s = " ".join(list(map(str,val)))
        cell_file.write(f"{len(val)}({s})\n")
    cell_file.write(")")
    cell_file.close()

if __name__ == "__main__":
    generateCellFile('owner', 'neighbour')
    generateBoundaryFile('boundary')