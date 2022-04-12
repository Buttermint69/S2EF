import ase.io
import ase.db.sqlite
import ase.io.trajectory
import numpy as np
import torch
from torch_geometric.data import Data
from itertools import product
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from tqdm import tqdm
import os
from embeddings import CONTINUOUS_EMBEDDINGS as c_embd, ATOMIC_RADII as a_rad

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

"""
try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm
"""



def collate(data_list):
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key])
            )
        elif isinstance(item[key], int) or isinstance(item[key], float):
            s = slices[key][-1] + 1
        else:
            raise ValueError("Unsupported attribute type")
        slices[key].append(s)

    if hasattr(data_list[0], "__num_nodes__"):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        if torch.is_tensor(data_list[0][key]):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key])
            )
        else:
            data[key] = torch.tensor(data[key])
        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices

def adjncy(Data_obj):
  sys_dim = Data_obj.natoms
  neigh_idx = Data_obj.edge_index[0]
  source_idx = Data_obj.edge_index[1]
  zeros = torch.zeros(sys_dim, sys_dim)
  for i in range(len(source_idx)):
    zeros[source_idx[i], neigh_idx[i]] = 1
  return zeros

class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.
        r_fixed (bool): Return a binary vector with flags for fixed (1) vs free (0) atoms.
        Default is True, so the fixed indices will be returned.

    """

    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        r_fixed=True,
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_fixed = r_fixed
        self.r_edges = r_edges

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        return _c_index, _n_index, n_distance, _offsets

    def _reshape_features(self, c_index, n_index, n_distance, offsets):
        """Stack center and neighbor index and reshapes distances,
        takes in np.arrays and returns torch tensors"""
        edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
        edge_distances = torch.FloatTensor(n_distance)
        cell_offsets = torch.LongTensor(offsets)

        # remove distances smaller than a tolerance ~ 0. The small tolerance is
        # needed to correct for pymatgen's neighbor_list returning self atoms
        # in a few edge cases.
        nonzero = torch.where(edge_distances >= 1e-8)[0]
        edge_index = edge_index[:, nonzero]
        edge_distances = edge_distances[nonzero]
        cell_offsets = cell_offsets[nonzero]

        return edge_index, edge_distances, cell_offsets

    def convert(
        self,
        atoms,
    ):
        """Convert a single atomic stucture to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        cell = torch.Tensor(np.array(atoms.get_cell())).view(1, 3, 3)
        natoms = positions.shape[0]
        atomic_masses = torch.Tensor(atoms.get_masses())
        
        
        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            atomic_masses = atomic_masses,
            natoms=natoms,
        )

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            split_idx_dist = self._get_neighbors_pymatgen(atoms)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.y = energy
        if self.r_forces:
            forces = torch.Tensor(atoms.get_forces(apply_constraint=False))
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if atoms.get_tags().any():
            tags = torch.Tensor(atoms.get_tags())
            data.tags = tags
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx
        #for i in tqdm(graph_data):
        
        x = torch.zeros((data.natoms ,14))
        for j in range(len(data.atomic_numbers)):
            emp = list()
            p  = list(data.pos[j].numpy())
            fix  = float(data.fixed[j].numpy())
            tag  = float(data.tags[j].numpy())
            embd = c_embd[int(data.atomic_numbers[j])]
            emp.extend(embd)
            emp.extend([tag, fix])
            emp.extend(p)
            
        x[j,...] = torch.tensor(emp)
        data.x = x
        
        arand = [0 if np.isnan(x) == True else x*0.01 for x in list(a_rad.values())]
        klo_edge_attr = torch.zeros(size = (len(data.distances),))
        for i in range(len(klo_edge_attr)):
            ed   = data.edge_index.t()[i][0]
            j_   =  data.edge_index.t()[i][1]
            sub  = (arand[int(data.atomic_numbers[int(ed)])]
                   + arand[int(data.atomic_numbers[int(j_)])]) 
            dis = data.distances[i] - sub
            if dis <0:
                dis = 1e-5
            else:
                pass
            klo_edge_attr[i] = dis
        data.distx = klo_edge_attr
        """
        zro = np.zeros(shape = (natoms,3,3))
        for i in range(natoms):
          zro[i, 0, ...] = [int(atomic_numbers[i]), float(atomic_masses[i]), int(tags[i])]
          zro[i, 1, ...] = positions[i,...].tolist()
          zro[i, 2, ...] = forces[i,...].tolist()
        
        data.x = torch.reshape(torch.Tensor(zro), 
                               shape=(natoms, 9))
        """
        return data

    def convert_all(
        self,
        atoms_collection,
        processed_file_path=None,
        collate_and_save=False,
        disable_tqdm=False,
    ):
        """Convert all atoms objects in a list or in an ase.db to graphs.

        Args:
            atoms_collection (list of ase.atoms.Atoms or ase.db.sqlite.SQLite3Database):
            Either a list of ASE atoms objects or an ASE database.
            processed_file_path (str):
            A string of the path to where the processed file will be written. Default is None.
            collate_and_save (bool): A boolean to collate and save or not. Default is False, so will not write a file.

        Returns:
            data_list (list of torch_geometric.data.Data):
            A list of torch geometric data objects containing molecular graph info and properties.
        """

        # list for all data
        data_list = []
        if isinstance(atoms_collection, list):
            atoms_iter = atoms_collection
        elif isinstance(atoms_collection, ase.db.sqlite.SQLite3Database):
            atoms_iter = atoms_collection.select()
        elif isinstance(
            atoms_collection, ase.io.trajectory.SlicedTrajectory
        ) or isinstance(atoms_collection, ase.io.trajectory.TrajectoryReader):
            atoms_iter = atoms_collection
        else:
            raise NotImplementedError

        for i , atoms in tqdm(enumerate(
            atoms_iter),
            desc="converting ASE atoms collection to graphs",
            total=len(atoms_collection),
            unit="systems",
            disable=disable_tqdm,
        ):
            # check if atoms is an ASE Atoms object this for the ase.db case
            if not isinstance(atoms, ase.atoms.Atoms):
                atoms = atoms.toatoms()
            data = self.convert(atoms)
            data_list.append(data)

            if collate_and_save:
                torch.save(data, os.path.join(processed_file_path,f'data_{i}.pt'))
        """
        if collate_and_save:
            data, slices = collate(data_list)
            torch.save((data, slices), processed_file_path)
        """
        return data_list
    
    

def mappings_(g, d):
  ls =  list(ase.symbols.Symbols(d.atomic_numbers))
  mappings = dict()
  counter = dict()
  for i in range(0,len(ls)):
    if ls[i] in counter.keys():

      counter[ls[i]] +=1
      mappings[i] = str(ls[i]) + str(counter[ls[i]])

    else:
      counter[ls[i]] = 0
      mappings[i] = str(ls[i]) + str(counter[ls[i]])
  return mappings


def add_attr(n_o_e, G, Data, attr):  
  # n_o_e - node or edge; for adding node attr: "n"
  #         for adding edge attr : "e"
  # G     - Networkx graph
  # Data  - torch geometrics Data object
  attr_ = dict()
  for i in range(Data.natoms):
    attr_[i] = Data[attr][i].numpy()
  if n_o_e == "n":
    nx.set_node_attributes(G, attr_, name = str(attr))
  elif n_o_e == "e":
    nx.set_edge_attributes(G, attr_, str(attr)) 



def network_plot_3D(G, angle, save=False, hide_axis = False):

  # Get node positions
  pos = nx.get_node_attributes(G, 'pos')
    
  # Get number of nodes
  n = G.number_of_nodes()

  # Get the maximum number of edges adjacent to a single node
  edge_max = max([G.degree(i) for i in range(n)])

  # Define color range proportional to number of edges adjacent to a single node
  colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)]
  matplotlib_axes_logger.setLevel('ERROR')
  # 3D network plot
  with plt.style.context(('ggplot')):
        
    fig = plt.figure(figsize=(20,20))
    ax = Axes3D(fig)
        
    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for key, value in pos.items():
      m_ = ase.symbols.Symbols([G.nodes[key]["atomic_numbers"]])  
      xi = value[0]
      yi = value[1]
      zi = value[2]
      
      # Scatter plot
      ax.scatter(xi, yi, zi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.5)
      ax.text(xi,yi,zi, str(m_), alpha = 1, fontsize = "xx-large", c = "green", ha = "center")
        
      # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
      # Those two points are the extrema of the line to be plotted
    for i,j in enumerate(G.edges()):

      x = np.array((pos[j[0]][0], pos[j[1]][0]))
      y = np.array((pos[j[0]][1], pos[j[1]][1]))
      z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
      # Plot the connecting lines
      ax.plot(x, y, z, c='black', alpha=0.1)
    
    # Set the initial view
    
    ax.view_init(30, angle)

    # Hide the axes
    if hide_axis:
        ax.set_axis_off()

  if save is not False:
    plt.savefig("/content/drive/My Drive/OCP_prac/graph_img" + str(angle).zfill(3)+ str(".png"))
    plt.close('all')
    pass
  else:
    plt.show()
  
  return


def edge_feat(data, disable_tqdm = False):
    
    """
    OPTIONAL
    Calculates edge attribute feature for all edge-node pairs.
    It parameterise the edges on the basis of distance from one another; 
    as the forces of neighbouring atom Atom2  far from the Atom1 have very 
    -little influence on Atom1 also vice-versa.
    The force influence is inversely propotional to the distances.
    f ~ 1/A;  f = forces, A = Distances
    f = h/A or h = f*A
   for Atom1 --> h1 = avg(forces)* int(disctances)
   for Atom2 --> h2 = avg(forces)* int(disctances)
   h = (h1 + h2)/2
    Args:
       list of Graphs in PyG data obj form.
    Returns:
        list containing edge_attr (H) for all graphs passed in Args.
        H[0] - edge_attr(h) for graph[0]; size(H[0]) == Number of edges in graph[0] 
    """
    
    H = list()
    
    for i in tqdm(range(0, len(data)), 
                  disable = disable_tqdm ):
        l = list()
        #dis = list()
        #force = list()
        data_ = data[i]
        z = np.zeros((data_.num_nodes, data_.num_nodes))
        
        for j in range(len(data_.distances)):
            z[data_.edge_index[0, j], data_.edge_index[1, j] ] = data_.distances[j]
            
        f_diag  = np.diag(np.average(data[i].force.numpy(), axis=-1))
        h_ =  f_diag@z
        h = (h_ + h_.T)*0.5
        for k in data_.edge_index.t():
            l.append(h[k[0], k[1]])
        
        H.append(l)
        #dis.append(data[i].distances.numpy())
        #force.append(diag)
    
    return H

