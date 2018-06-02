//Charlie Mueller 2/24/2016
#include <iostream>
#include <fstream>
#include "TSystem.h"
#include <vector>
#include "TH1.h"
#include "TChain.h"
#include <string>
#include <algorithm>
#include "TString.h"
#include "TH1D.h"
#include "TFile.h"
#include <cmath>
#include "TLorentzVector.h"
#include "ttH-13TeVMultiLeptons/TemplateMakers/src/classes.h"
#include "TMVA/Config.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"
#include "TMVA/MethodCuts.h"
#include "selection.h"
#include "loadSamples.h"
#include "treeTools.h"
#include "TCanvas.h"
#include "TH2I.h"
#include "DataFormats/Math/interface/deltaR.h"

/////////////////////////////////////////
///
/// usage: root -l makeSelectionTrees.C+
///
/////////////////////////////////////////

struct reverse_probb_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return ((struct1.DeepCSVprobb + struct1.DeepCSVprobbb) > (struct2.DeepCSVprobb + struct2.DeepCSVprobbb));
    }
};

struct pt_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return (struct1.obj.pt() < struct2.obj.pt());
    }
};

void write_csv(std::ofstream& in_file, vector<ttH::Jet> in_jets, vector<int> indices)
{  
  int size = indices.size();
  for (int i=0; i<size; i++) {
    in_file << indices[i];
    if (i<(size - 1))
        in_file << ".";
    else
        in_file << ",";
  }

  sort(in_jets.begin(), in_jets.end(), reverse_probb_key());
  ttH::Jet bjet = in_jets.back();
  in_file<< bjet.obj.pt() << "," bjet.obj.M() <<",";
  in_file<< (bjet.DeepCSVprobb + bjet.DeepCSVprobbb) <<","<< bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobudsg) <<",";
  in_file<< bjet.DeepCSVprobc/(bjet.DeepCSVprobc + bjet.DeepCSVprobb + bjet.DeepCSVprobbb) <<",";
  in_file<< bjet.ptD <<","<< bjet.axis1 <<","<< bjet.mult <<",";
  
  in_jets.pop_back();
  sort(in_jets.begin(), in_jets.end(), pt_key());

  for (const auto & jet : in_jets)
    {
      in_file<< jet.obj.pt() << "," jet.obj.M() <<",";
      in_file<< (jet.DeepCSVprobb + jet.DeepCSVprobbb) <<","<< jet.DeepCSVprobc/(jet.DeepCSVprobc + jet.DeepCSVprobudsg) <<",";
      in_file<< jet.DeepCSVprobc/(jet.DeepCSVprobc + jet.DeepCSVprobb + jet.DeepCSVprobbb) <<",";
      in_file<< jet.ptD <<","<< jet.axis1 <<","<< jet.mult <<",";
    }
  
  in_file<<"\n";
  in_file.flush();
}

void run_it(TChain* tree, TString output_file)
{

  //int num_hadronic = 0;
  int treeentries = tree->GetEntries();   
  cout << "# events in tree: "<< treeentries << endl;
  int passed_selection = 0;

  double mcwgt_intree = -999.;
  double wallTimePerEvent_intree = -99.;  
  int eventnum_intree = -999;
  int lumiBlock_intree = -999;
  int runNumber_intree = -999;
  
  vector<ttH::Jet> *preselected_jets_intree=0;
  vector<ttH::GenParticle> *gen_parts=0;
  
  tree->SetBranchStatus("*",0);
  tree->SetBranchStatus("mcwgt",1);
  tree->SetBranchStatus("wallTimePerEvent",1);
  tree->SetBranchStatus("eventnum",1);
  tree->SetBranchStatus("lumiBlock",1);
  tree->SetBranchStatus("preselected_jets.*",1);
  tree->SetBranchStatus("pruned_genParticles.*",1);


  tree->SetBranchAddress("mcwgt", &mcwgt_intree);
  tree->SetBranchAddress("wallTimePerEvent", &wallTimePerEvent_intree);
  tree->SetBranchAddress("eventnum", &eventnum_intree);
  tree->SetBranchAddress("lumiBlock", &lumiBlock_intree);
  tree->SetBranchAddress("preselected_jets", &preselected_jets_intree);
  tree->SetBranchAddress("pruned_genParticles", &gen_parts);
  
  ofstream outfile;
  outfile.open(output_file);

  Int_t cachesize = 500000000;   //500 MBytes
  tree->SetCacheSize(cachesize);
  tree->SetCacheLearnEntries(20); 


  double starttime = get_wall_time();
  //  treeentries = 1000000;
  int num_signal = 0;
  int num_bkgd = 0;
  float avg_bkgd = 0;
  for (int i=0; i<treeentries; i++)
    {

      //printProgress(i,treeentries);
      tree->GetEntry(i);
      
      cout<< "Finished: " <<(i+1)*100/treeentries <<"%\r";

      if (preselected_jets_intree->size() < 3) continue;
      vector<ttH::Jet> matched_jets[2] = {vector<ttH::Jet>(), vector<ttH::Jet>()};
      vector<int> matched_ix[2] = {vector<int>(), vector<int>()};
      vector<ttH::Jet> unmatched_jets;
      int ix = 0;
      int num_pos_lept = 0;
      int num_neg_lept = 0;
      bool tau_lept = false;
      for (const auto &pjet : *preselected_jets_intree) {
        int ID = pjet.genPdgID;
        int mID = pjet.genMotherPdgID;
        int gmID = pjet.genGrandMotherPdgID;
        if (ID == 5 && mID == 6) {
          matched_jets[0].insert(matched_jets[0].begin(), pjet);
          matched_ix[0].push_back(ix);
        } else if (ID == -5 && mID == -6) {
          matched_jets[1].insert(matched_jets[1].begin(), pjet);
          matched_ix[1].push_back(ix);
        } else if (mID == 24 && gmID == 6) {
          matched_jets[0].push_back(pjet);
          matched_ix[0].push_back(ix);
        } else if (mID == -24 && gmID == -6) {
          matched_jets[1].push_back(pjet);
          matched_ix[1].push_back(ix);
        } else if (abs(ID) == 15) {
          tau_lept = true;
        } else if (ID >= 11 && ID <= 16 && (mID == 24 || gmID == 24)) {
          num_pos_lept++;  
        } else if (ID <= -11 && ID >= -16 && (mID == -24 || gmID == -24)) {
          num_neg_lept++;
        } else {
          unmatched_jets.push_back(pjet);
        }
        ix++;
      }
      
      // Make sure there's only one hadronic top
      int size = preselected_jets_intree->size();
      bool matched1 = (matched_jets[0].size() == 3);
      bool matched2 = (matched_jets[1].size() == 3);
      if (matched1 == matched2 || num_pos_lept < 2 || num_neg_lept < 2 || !tau_lept)
        continue;
      
      passed_selection++;
          
      // Generate all combinations of indices via 3 for loops
      // Every iteration make a vector to compare against matched vectors if there is a fully matched triplet
      // Write the triplet to file
      for (int i=0; i < size-2; i++) {
        for (int j=i+1; j < size-1; j++) {
          for (int k=j+1; k < size; k++) {
            vector<int> comb = {i, j, k};
            if (matched1 && (comb == matched_ix[0])) {
                write_csv(outfile, *preselected_jets_intree, comb);
            } else if (matched2 && (comb == matched_ix[1])) {
                write_csv(outfile, *preselected_jets_intree, comb);
            }
          }
        }
      }
    }

  double endtime = get_wall_time();
  cout << "Elapsed time: " << endtime - starttime << " seconds, " << endl;
  if (treeentries>0) cout << "an average of " << (endtime - starttime) / treeentries << " per event." << endl;
  cout << "Number that passed selection: " <<passed_selection<< endl;
  cout << "Num Sig: " <<num_signal<< endl;
  cout << "Num Bkgd: " <<num_bkgd<< endl;
  cout << "Avg Bkgd: " <<avg_bkgd<< endl;
  
  outfile.close();
}

void makeTagTripletsSet(TString sample="")
{

  TString output_dir = "";
  TString input_files[7] = {"trees/output_tree_15290.root", "trees/output_tree_18889.root",
                            "trees/output_tree_25083.root", "trees/output_tree_21531.root",
                            "trees/output_tree_18046.root", "trees/output_tree_69439.root",
                            "trees/output_tree_39898.root"};
  
  sample = "ttH";
  TString output_file = output_dir + sample + "_triplets.csv";
  TChain *tth_chain = new TChain("OSTwoLepAna/summaryTree");    
  for (int i=0; i<7; i++) {
    tth_chain->Add(input_files[i]);
  }

  run_it(tth_chain, output_file);

}
