"""Mutation Generator (MuGen) class
Written by Ehsan Motazedi, 14-08-2015, Wageningen UR"""
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Seq import MutableSeq
from Bio.Alphabet import *
import copy
import random
import sys


class CustomException(Exception):
	def __init__(self, value):
		self.parameter = value

	def __str__(self):
		return repr(self.parameter)


class MuGen(object):
	""" performs mutations and deletion/insertion with desired porbability
	and desired structure. Gets a Seq object, a mutation or indel dicitonary,
	and the probablities for each item in those dictionaries.
	insertprob and deleteprob are base specefic probabilities of length 4
	mualphabet is a dictionary specifying the possible mutations for each letter of
	the sequence alphabet.
	muprob gives the mutation probality for each letter of the sequence alphabet."""

	def __init__(self, seq, alphaproperty=None, insertprob=None,
		     deleteprob=None, mualphabet=None,
		     muprob=None, mupos=None, delpos=None, inpos=None,
		     verbose=False):
		try:
			self.occureddel = list()  # This is to keep a history of chnges made to the reference
			self.occuredmu = list()  # This is necessary for writing the haplotypes in the format
			self.occuredins = list()  # of haplotyping software's.
			self.inserted_allele = list()  # keeps track of the inserted allele to be able to get them back when needed!
			self.alt_allele = list()  # keeps track of the substituted
			if not isinstance(verbose, bool):
				raise CustomException("ERROR: verbose must be set to either True or False. \
Default is to False")
			else:
				self.verbose = verbose
			if isinstance(seq, str):
				if alphaproperty is None:
					if self.verbose:
						print(
							"WARNING: No alphabet type is specified for the sequence string!")
					else:
						pass
					self.alphaproperty = Alphabet()
				else:
					self.alphaproperty = alphaproperty
				self.seq = MutableSeq(seq, self.alphaproperty)
			elif isinstance(seq, Seq):
				self.alphaproperty = seq.__getattribute__(
					'alphabet')
				self.seq = seq.tomutable()
			elif isinstance(seq, MutableSeq):
				self.alphaproperty = seq.__getattribute__(
					'alphabet')
				self.seq = copy.deepcopy(seq)
			else:
				raise CustomException("ERROR: Should provide a Seq or MutableSeq object, \n \
or a string sequence!")
			self.alphabet = set(str(self.seq))
			self.ref = str(self.seq)
			if not delpos:
				self.delpos = []
			else:
				if set(delpos).issubset(
					set(range(len(self.ref)))):
					self.delpos = list(
						delpos)  # Deletion by specifying the positions
				else:
					raise CustomException(
						"ERROR: Deletion positions exceed the range of the reference or are not positive integers!")
			if not inpos:
				self.inpos = []
			else:
				if set(inpos).issubset(
					set(range(len(self.ref)))):
					self.inpos = list(
						inpos)  # Insertion by specifying the positions
				else:
					raise CustomException(
						"ERROR: Insertion positions exceed the range of the reference or are not positive integers!")
			if not mupos:
				self.mupos = []
			else:
				if set(mupos).issubset(
					set(range(len(self.ref)))):
					self.mupos = list(
						mupos)  # Mutation by specifying the positions
				else:
					raise CustomException(
						"ERROR: Mutation positions exceed the range of the reference or are not positive integers!")
			if not mualphabet:
				if self.verbose:
					print("WARNING: You have specified no mutation alphabet! Mutations are set to random \
letters!")
				self.mualphabet = dict()
				for key in self.alphabet:
					self.mualphabet[key] = ''.join(
						self.alphabet - {
						key})  # Non-specified mutations could happen to any letter
			else:
				mualphabet = dict([(str(k), str(v)) for k, v in
						   mualphabet.iteritems()])
				for key, value in mualphabet.iteritems():
					if len(key) != 1:
						raise CustomException("ERROR: the mutation alphabet deals with point mutations! Only single letters are\
 allowed as keys!")
					elif key in set(''.join(value)):
						raise CustomException("ERROR: Wrong mutation values specified! A letter could just be substituted with a\
 different letter for mutation!")
				if set(
					mualphabet.keys()) == self.alphabet and set(
					''.join(
						mualphabet.values())) <= self.alphabet:
					self.mualphabet = copy.deepcopy(
						mualphabet)
				elif set(
					mualphabet.keys()) < self.alphabet and set(
					''.join(
						mualphabet.values())) < self.alphabet:
					if self.verbose:
						print("WARNING: Mutation is not specified for some letters! Those mutations are set\
 to random letters!")
					self.mualphabet = copy.deepcopy(
						mualphabet)  # Whatever has been specified for mutation alphabet is kep intact
					for key in self.alphabet - set(
						mualphabet.keys()):
						self.mualphabet[key] = ''.join(
							self.alphabet - {
							key})  # Non-specified mutations could happen to any letter
				else:
					if self.verbose:
						print("WARNING: Mutation alphabet is not compatible with sequence alphabet! Both alphabets are\
 updated and\nunspecified mutations are set to random letters!")
					new_mualphabet = dict()  # As mutation may introduce novel alleles in the sequence, alphabet is updated first
					for key, value in mualphabet.iteritems():  # Whatever has been specified for mutation alphabet is kep intact
						self.alphabet.add(
							key)  # Only the alphabet is updated if necessary
						self.alphabet |= (set(''.join(
							value)) - self.alphabet)
						new_mualphabet.update(
							{key: value})
					for key in self.alphabet - set(
						new_mualphabet.keys()):
						new_mualphabet[key] = ''.join(
							self.alphabet - {
							key})  # Non-specified mutations could happen to any letter
					self.mualphabet = copy.deepcopy(
						new_mualphabet)
			if not insertprob:
				self.insertprob = dict()  # If no insertprob is given, it is set to zero everywhere
				for key in self.alphabet:
					self.insertprob[key] = 0
			else:
				if set(list(
					insertprob.keys())) != self.alphabet:
					if self.verbose:
						print("WARNING: Missing/Invalid letter(s) in insertion probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_insertprob = dict()
				for key, value in insertprob.iteritems():
					if value >= 0 and value <= 1:
						new_insertprob.update(
							{key: value})
					else:
						raise CustomException(
							"ERROR: Insertion probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_insertprob.keys()):
					new_insertprob[key] = 0
				self.insertprob = copy.deepcopy(new_insertprob)
			if not deleteprob:  # If no deleteprob is given, it is set to zero everywhere
				self.deleteprob = dict()
				for key in self.alphabet:
					self.deleteprob[key] = 0
			else:
				if set(list(
					deleteprob.keys())) != self.alphabet:
					if self.verbose:
						print("WARNING: Missing/Invalid letter(s) in deletion probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_deleteprob = dict()
				for key, value in deleteprob.iteritems():
					if value >= 0 and value <= 1:
						new_deleteprob.update(
							{key: value})
					else:
						raise CustomException(
							"ERROR: Deletion probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_deleteprob.keys()):
					new_deleteprob[key] = 0
				self.deleteprob = copy.deepcopy(new_deleteprob)
			if not muprob:
				self.muprob = dict()  # If no muprob is given, it is set to zero everywhere
				for key in self.alphabet:
					self.muprob[key] = 0
			else:
				if set(list(muprob.keys())) != self.alphabet:
					if self.verbose:
						print("WARNING: Missing/Invalid letter(s) in mutation probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_muprob = dict()
				for key, value in muprob.iteritems():
					if value >= 0 and value <= 1:
						new_muprob.update({key: value})
					else:
						raise CustomException(
							"ERROR: Mutation probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_muprob.keys()):
					new_muprob[key] = 0
				self.muprob = copy.deepcopy(new_muprob)
		except CustomException as instance:
			print(instance)
			sys.exit(2)
		else:
			if self.verbose:
				print(
					"MuGen object successfully created.\nWARNING: MuGen sequence is case sensitive!")

	def __repr__(self):
		return "Haplotype: %s, \n Reference sequence: %s, \n Mutation probabilty: %s, \n Mutations: %s, \n \
Insertion probabilty: %s, \n Deletion Probability: %s, \n \
Insertion positions: %s, \n Deletion positions: %s, \n Mutation positions: %s \n" % (
		self.seq, self.ref,
		self.muprob, self.mualphabet, self.insertprob, self.deleteprob,
		self.inpos, self.delpos, self.mupos)

	def __str__(self):
		return repr(self)

	def get_hap(self):  # Access Methods
		return self.seq

	def get_ref(self):
		return self.ref

	def get_insertprob(self):
		return self.insertprob

	def get_deleteprob(self):
		return self.deleteprob

	def get_muprob(self):
		return self.muprob

	def get_mualphabet(self):
		return self.mualphabet

	def get_mupos(self):
		return self.mupos

	def get_inpos(self):
		return self.inpos

	def get_delpos(self):
		return self.delpos

	def get_occureddelpos(self):
		return self.occureddel

	def get_occuredmupos(self):
		return self.occuredmu

	def get_occuredinspos(self):
		return self.occuredins

	def get_ins_allele(self):
		return self.inserted_allele

	def get_mu_allele(self):
		return self.alt_allele

	def set_ref(self, ref):  # Modifier methods
		"""Changes the reference sequence of the MuGen object. Could become problematic if the new reference
		has a different length than the current reference, while indel and mutation positions are specified.
		A useful method if reference is a mutable seq entity which is constantly called and changed by other
		methods and calsses."""
		try:
			if set(str(ref)).issubset(self.alphabet):
				if not set(self.mupos).issubset(
					set(range(len(str(ref))))):
					raise CustomException(
						"ERROR: Mutation positions exceed the range of the new reference!")
				elif not set(self.inpos).issubset(
					set(range(len(str(ref))))):
					raise CustomException(
						"ERROR: Insertion positions exceed the range of the new reference!")
				elif not set(self.delpos).issubset(
					set(range(len(str(ref))))):
					raise CustomException(
						"ERROR: Deletion positions exceed the range of the new reference!")
				else:
					self.ref = str(ref)
			else:
				raise CustomException(
					"ERROR: the new reference is not compatible with the current alphabet!")
		except CustomException as instance:
			print("Failed to update the reference!")
			print(instance)
		except:
			print("Failed to update the reference!")
			raise
		else:
			if self.verbose:
				print(
					"The reference sequence has been updated!")

	def set_pos(self, inpos=None, delpos=None, mupos=None, ):
		"""Changes the insertion, deletion and substitution sites of the MuGen object. A useful method if
		posmu and probmu methods are constantly called."""
		try:
			changedel = 0  # If set to 1, delpos is changed. Otherwise no change to delpos.
			changein = 0  # If set to 1, inpos is changed. Otherwise no change to inpos.
			changemu = 0  # If set to 1, mupos is changed. Otherwise no change to mupos.
			if delpos is None:  # Default is no change
				pass
			else:
				if set(delpos).issubset(
					set(range(len(self.ref)))):
					changedel = 1
				else:
					raise CustomException(
						"ERROR: New deletion positions exceed the range of the reference or are not positive integers!")
			if inpos is None:  # Deafult is no change
				pass
			else:
				if set(inpos).issubset(
					set(range(len(self.ref)))):
					changein = 1
				else:
					raise CustomException(
						"ERROR: New insertion positions exceed the range of the reference or are not positive integers!")
			if mupos is None:  # Default is no change
				pass
			else:
				if set(mupos).issubset(
					set(range(len(self.ref)))):
					changemu = 1
				else:
					raise CustomException(
						"ERROR: New mutation positions exceed the range of the reference or are not positive integers!")
			if changedel:
				self.delpos = list(delpos)  # Update delpos
			else:
				pass
			if changein:
				self.inpos = list(inpos)  # Update inpos
			else:
				pass
			if changemu:
				self.mupos = list(mupos)  # Update mupos
			else:
				pass
		except CustomException as instance:
			print("Failed to update indel and mutation positions!")
			print(instance)
		except:
			print("Failed to update indel and mutation positions!")
			raise
		else:
			if self.verbose:
				print("Indel and mutation positions updated!")

	def set_prob(self, insertprob=None, deleteprob=None, muprob=None):
		"""Changes the insertion, deletion and mutation probabilities of the MuGen object. A useful method if
		posmu and probmu methods are constantly called."""
		try:
			noinsert = -1
			nodel = -1
			nomu = -1
			if insertprob is None:  # Default to no change
				noinsert = 0
			elif not insertprob:
				noinsert = 1
			elif set(list(insertprob.keys())) != self.alphabet:
				if self.verbose:
					print("WARNING: Missing/Invalid letter(s) in insertion probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_insertprob = dict()
				for key, value in insertprob.iteritems():
					if value >= 0 and value <= 1:
						new_insertprob.update(
							{key: value})
					else:
						raise CustomException(
							"ERROR: Insertion probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_insertprob.keys()):
					new_insertprob[key] = 0
			else:
				new_insertprob = copy.deepcopy(insertprob)
			if deleteprob is None:  # Default to no change
				nodel = 0
			elif not deleteprob:  # If empty deleteprob is given, it is set to zero everywhere
				nodel = 1
			elif set(list(deleteprob.keys())) != self.alphabet:
				if self.verbose:
					print("WARNING: Missing/Invalid letter(s) in deletion probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_deleteprob = dict()
				for key, value in deleteprob.iteritems():
					if value >= 0 and value <= 1:
						new_deleteprob.update(
							{key: value})
					else:
						raise CustomException(
							"ERROR: Deletion probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_deleteprob.keys()):
					new_deleteprob[key] = 0
			else:
				new_deleteprob = copy.deepcopy(deleteprob)
			if muprob is None:  # Default to no change
				nomu = 0
			elif not muprob:
				nomu = 1
			elif set(list(muprob.keys())) != self.alphabet:
				if self.verbose:
					print("WARNING: Missing/Invalid letter(s) in mutation probability!\n\
Probabilities are set to zero for missing letters! Invalid letters are ignored!")
				new_muprob = dict()
				for key, value in muprob.iteritems():
					if value >= 0 and value <= 1:
						new_muprob.update({key: value})
					else:
						raise CustomException(
							"ERROR: Mutation probability must be >=0 and <=1!")
				for key in self.alphabet - set(
					new_muprob.keys()):
					new_muprob[key] = 0
			else:
				new_muprob = copy.deepcopy(muprob)
			if nodel == 0:
				pass
			elif nodel == 1:
				self.deleteprob = dict()
				for key in self.alphabet:
					self.deleteprob[key] = 0
			else:
				self.deleteprob = copy.deepcopy(
					new_deleteprob)  # Update deleteprob
			if nomu == 0:
				pass
			elif nomu == 1:
				self.muprob = dict()  # If empty muprob is given, it is set to zero everywhere
				for key in self.alphabet:
					self.muprob[key] = 0
			else:
				self.muprob = copy.deepcopy(
					new_muprob)  # Update muprob
			if noinsert == 0:
				pass
			elif noinsert == 1:
				self.insertprob = dict()  # If empty insertprob is given, it is set to zero everywhere
				for key in self.alphabet:
					self.insertprob[key] = 0
			else:
				self.insertprob = copy.deepcopy(
					new_insertprob)  # Update insertprob
		except CustomException as instance:
			print(instance)
			print(
				"Failed to update indel and mutation probabilities!")
		except:
			print(
				"Failed to update indel and mutation probabilities!")
			raise
		else:
			if self.verbose:
				print(
					"Indel and mutation probabilities successfully updated!")

	def set_mualphabet(self, mualphabet=None):
		"""Changes the mutation alphabet of the MuGen object. A useful method if posmu and probmu methods
		are constantly called."""
		try:
			if not mualphabet:
				if self.verbose:
					print("WARNING: You have specified no mutation alphabet! Mutations are set to random \
letters!")
				self.mualphabet = dict()
				for key in self.alphabet:
					self.mualphabet[key] = ''.join(
						self.alphabet - {
						key})  # Non-specified mutations could happen to any letter
			else:
				mualphabet = dict([(str(k), str(v)) for k, v in
						   mualphabet.iteritems()])
				for key, value in mualphabet.iteritems():
					if len(key) != 1:
						raise CustomException("ERROR: the mutation alphabet deals with point mutations! Only single letters are\
 allowed as keys!")
					elif key in set(''.join(value)):
						raise CustomException("ERROR: Wrong mutation values specified! A letter could just be substituted with a\
 different letter for mutation!")
				if set(
					mualphabet.keys()) == self.alphabet and set(
					''.join(
						mualphabet.values())) <= self.alphabet:
					self.mualphabet = copy.deepcopy(
						mualphabet)
				elif set(
					mualphabet.keys()) < self.alphabet and set(
					''.join(
						mualphabet.values())) < self.alphabet:
					if self.verbose:
						print("WARNING: Mutation is not specified for some letters! Those mutations are set\
 to random letters!")
					self.mualphabet = copy.deepcopy(
						mualphabet)  # Whatever has been specified for mutation alphabet is kep intact
					for key in self.alphabet - set(
						mualphabet.keys()):
						self.mualphabet[key] = ''.join(
							self.alphabet - {
							key})  # Non-specified mutations could happen to any letter
				else:
					if self.verbose:
						print("WARNING: Mutation alphabet is not compatible with sequence alphabet! Both alphabets are\
 updated and\nunspecified mutations are set to random letters!")
					new_mualphabet = dict()  # As mutation may introduce novel alleles in the sequence, alphabet is updated first
					for key, value in mualphabet.iteritems():  # Whatever has been specified for mutation alphabet is kep intact
						self.alphabet.add(
							key)  # Only the alphabet is updated if necessary
						self.alphabet |= (set(''.join(
							value)) - self.alphabet)
						new_mualphabet.update(
							{key: value})
					for key in self.alphabet - set(
						new_mualphabet.keys()):
						new_mualphabet[key] = ''.join(
							self.alphabet - {
							key})  # Non-specified mutations could happen to any letter
					self.mualphabet = copy.deepcopy(
						new_mualphabet)

		except CustomException as instance:
			print(instance)
			print("Mualphabet could not be updated!")
		except:
			print("Mualphabet could not be updated!")
			raise
		else:
			if self.verbose:
				print("Mualphabet successfully updated!")

	def probmu(self):
		self.occuredmu = list()
		self.occureddel = list()
		self.occuredins = list()
		self.inserted_allele = list()
		self.alt_allele = list()
		"""Operates on a MuGen object, and returns a Seq object obtained by making random changes
		to the reference sequence of the MuGen object, using the probabilities given to MuGen"""
		self.seq = []
		for __site, __base in enumerate(self.ref):
			if __site in set(self.mupos) | set(self.inpos) | set(
				self.delpos):
				self.seq.append(
					__base)  # No change is made at indel/mutation positions
			else:
				__prob = {'ins': self.insertprob.get(__base),
					  'del': self.deleteprob.get(__base),
					  'sub': self.muprob.get(__base)}
				__error = random.choice(['ins', 'del', 'sub',
							 'sub'])  # An error occurs randomly: insertion or \
				# deletion or substitution
				__rnd = float(int(
					random.random() * 100000)) / 100000  # The probability that this error is \
				# not corrected by replication machinary is determined \
				if __rnd < __prob.get(
					__error):  # by insertprob,deleteprob and muprob
					if __error == 'sub':
						self.seq.append(random.choice(
							self.mualphabet.get(
								__base)))  # Substitute tha letter with one from the mutation alphabet
						self.occuredmu.append(
							__site)  # Update the list of the sites where a mutation has occured
						self.alt_allele.extend([
									       self.seq[
										       -1]])  # Update the list of alternative alleles
					elif __error == 'ins':
						self.seq.append(__base)
						self.seq.append(random.choice(
							list(
								self.alphabet)))  # Insert a random letter right after the letter
						self.occuredins.append(
							__site)  # Update the list of the sites after which an insertion has occured
						self.inserted_allele.extend([
										    __base +
										    self.seq[
											    -1]])  # Update the list of inserted alleles
					else:
						self.occureddel.append(
							__site)  # Delete the letter in the progeny sequence by just not adding it
				else:  # Update the list of the sites which are deleted in the progeny sequence
					self.seq.append(
						__base)  # No change is induced at the site in the progeny sequence
		self.seq = ''.join(self.seq)
		self.seq = MutableSeq(self.seq, self.alphaproperty)
		if (self.occuredins):
			_ins_allele = zip(self.occuredins,
					  self.inserted_allele)
			_ins_allele.sort(key=lambda tup: tup[
				0])  # Sort the occured change positions in ascending order
			self.occuredins, self.inserted_allele = zip(
				*_ins_allele)
			self.occuredins = list(self.occuredins)
			self.inserted_allele = list(self.inserted_allele)
			_ins_allele = None
		else:
			self.inserted_allele = []
			self.occuredins = []
		if (self.occuredmu):
			_alt_allele = zip(self.occuredmu, self.alt_allele)
			_alt_allele.sort(key=lambda tup: tup[0])
			self.occuredmu, self.alt_allele = zip(*_alt_allele)
			self.occuredmu = list(self.occuredmu)
			self.alt_allele = list(self.alt_allele)
			_alt_allele = None
		else:
			self.occuredmu = []
			self.alt_allele = []
		if (self.occureddel):
			self.occureddel.sort()
		else:
			self.occureddel = []
		if self.verbose:
			print("WARNING: If indel/mutation positions are specified, MuGen.probmu() makes no change at those sites. \n \
Use MuGen.posmu() or Mugen.hapchanger() to apply changes at those sites!")
			print("Changes made to the haplotype!")

	def posmu(self):
		"""Operates on a MuGen object, and returns a Seq object obtained by making specefic changes
		at specefic locations on the reference sequence of the MuGen object, using the
		indel and mutation positions already given to MuGen"""
		__change = [None] * len(self.ref)
		self.occuredmu = list()
		self.occureddel = list()
		self.occuredins = list()
		self.inserted_allele = list()  # Preservation and change site are determined
		self.alt_allele = list()
		for __site in self.inpos:  # Preservation and change site are determined
			__change[
				__site] = 'ins'  # with respect to the reference seq
		for __site in self.delpos:  # type of the change is also specified
			__change[__site] = 'del'  # The substituion base at the
		for __site in self.mupos:  # specified position is determined
			__change[__site] = 'sub'  # from the mutation alphabet.
		self.seq = []
		for __site, __error in iter(
			zip(range(len(self.ref)), __change)):
			__base = self.ref[__site]
			if __error is None:
				self.seq.append(__base)
			elif __error == 'sub':
				self.seq.append(random.choice(
					self.mualphabet.get(
						__base)))  # Substitute tha letter with one from the mutation alphabet
				self.occuredmu.append(
					__site)  # Update the list of the sites where a mutation has occured
				self.alt_allele.extend([self.seq[
								-1]])  # Update the list of alternative alleles
			elif __error == 'ins':
				self.seq.append(__base)
				self.seq.append(random.choice(list(
					self.alphabet)))  # Insert a random letter right after the letter
				self.occuredins.append(
					__site)  # Update the list of the sites after which an insertion has occured
				self.inserted_allele.extend([__base + self.seq[
					-1]])  # Update the list of inserted alleles
			else:
				self.occureddel.append(
					__site)  # Delete the letter in the progeny sequence by just not adding it
		self.seq = ''.join(self.seq)
		self.seq = MutableSeq(self.seq,
				      self.alphaproperty)  # Update the list of the sites which are deleted in the progeny sequence
		if self.occuredins:
			_ins_allele = zip(self.occuredins,
					  self.inserted_allele)
			_ins_allele.sort(key=lambda tup: tup[
				0])  # Sort the occured change positions
			self.occuredins, self.inserted_allele = zip(
				*_ins_allele)
			self.occuredins = list(self.occuredins)
			self.inserted_allele = list(self.inserted_allele)
			_ins_allele = None
		else:
			self.inserted_allele = []
			self.occuredins = []
		if (self.occuredmu):
			_alt_allele = zip(self.occuredmu, self.alt_allele)
			_alt_allele.sort(key=lambda tup: tup[0])
			self.occuredmu, self.alt_allele = zip(*_alt_allele)
			self.occuredmu = list(self.occuredmu)
			self.alt_allele = list(self.alt_allele)
			_alt_allele = None
		else:
			self.occuredmu = []
			self.alt_allele = []
		if (self.occureddel):
			self.occureddel.sort()
		else:
			self.occureddel = []
		if self.verbose:
			print("WARNING: if there are overlaps betweeen deletion, insertion and mutation positions, \n \
just one of the changes takes place with the following priority: \n \
1)Mutation  2)Deletion 3)Insertion. \n")
			print("Changes made to the haplotype!")

	def hapchanger(self):
		"""Operates on a MuGen object, and returns a Seq object obtained by making random and specified
		changes to the reference sequence of the MuGen object, using the probabilities as well as the
		positions given to MuGen."""
		self.seq = []
		self.occuredmu = list()
		self.occureddel = list()
		self.occuredins = list()
		self.inserted_allele = list()
		self.alt_allele = list()
		for __site, __base in enumerate(self.ref):
			if __site in set(
				self.mupos):  # Making specified changes at the specified positions
				self.seq.append(random.choice(
					self.mualphabet.get(
						__base)))  # Induce mutation at the site whose position is given
				self.occuredmu.append(
					__site)  # Update the list of the sites where a mutation has occured
				self.alt_allele.extend([self.seq[
								-1]])  # Update the list of alternative alleles
			elif __site in set(self.inpos):
				self.seq.append(
					__base)  # Make an insertion right after the site whose position is given
				self.seq.append(
					random.choice(list(self.alphabet)))
				self.occuredins.append(
					__site)  # Update the list of the sites after which an insertion has occured
				self.inserted_allele.extend([__base + self.seq[
					-1]])  # Update the list of inserted alleles
			elif __site in set(self.delpos):
				self.occureddel.append(
					__site)  # Update the list of the sited with deleted letter
			else:  # If not change is specified at the position, \
				# make a random change according to the prob model
				__prob = {'ins': self.insertprob.get(__base),
					  'del': self.deleteprob.get(__base),
					  'sub': self.muprob.get(__base)}
				__error = random.choice(['ins', 'del', 'sub',
							 'sub'])  # An error occurs randomly: insertion or \
				# deletion or substitution
				__rnd = float(int(
					random.random() * 100000)) / 100000  # The probability that this error is \
				# not corrected by replication machinary is determined \
				if __rnd < __prob.get(
					__error):  # by insertprob,deleteprob and muprob
					if __error == 'sub':
						self.seq.append(random.choice(self.mualphabet.get(__base)))
						self.occuredmu.append(__site)  # Update the list of the sites where a mutation has occured
						self.alt_allele.extend([self.seq[-1]])  # Update the list of alternative alleles
					elif __error == 'ins':
						self.seq.append(__base)
						self.seq.append(random.choice(list(self.alphabet)))
						self.occuredins.append(__site)  # Update the list of the sites after which an insertion has occured
						self.inserted_allele.extend([__base + self.seq[-1]])  # Update the list of inserted alleles
					elif __error == 'del':
						self.occureddel.append(__site)  # Update the list of the sited with deleted letter
				else:
					self.seq.append(__base)
		self.seq = ''.join(self.seq)
		self.seq = MutableSeq(self.seq, self.alphaproperty)
		if (self.occuredins):
			_ins_allele = zip(self.occuredins,
					  self.inserted_allele)
			_ins_allele.sort(key=lambda tup: tup[
				0])  # Sort the occured change positions
			self.occuredins, self.inserted_allele = zip(
				*_ins_allele)
			self.occuredins = list(self.occuredins)
			self.inserted_allele = list(self.inserted_allele)
			_ins_allele = None
		else:
			self.inserted_allele = []
			self.occuredins = []
		if (self.occuredmu):
			_alt_allele = zip(self.occuredmu, self.alt_allele)
			_alt_allele.sort(key=lambda tup: tup[0])
			self.occuredmu, self.alt_allele = zip(*_alt_allele)
			self.occuredmu = list(self.occuredmu)
			self.alt_allele = list(self.alt_allele)
			_alt_allele = None
		else:
			self.occuredmu = []
			self.alt_allele = []
		if (self.occureddel):
			self.occureddel.sort()
		else:
			self.occureddel = []
		if self.verbose:
			print("Changes made to the haplotype!")