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
#include <dirent.h>

/////////////////////////////////////////
///
/// usage: root -l makeSelectionTrees.C+
///
/////////////////////////////////////////

struct less_than_key
{
    inline bool operator() (const ttH::Jet& struct1, const ttH::Jet& struct2)
    {
        return ((struct1.DeepCSVprobb + struct1.DeepCSVprobbb) < (struct2.DeepCSVprobb + struct2.DeepCSVprobbb));
    }
};

void write_csv(std::ofstream& in_file, vector<ttH::Jet> in_jets, int event_num, int sig_or_bkgd)
{
  in_file << sig_or_bkgd << ",";  

  sort(in_jets.begin(), in_jets.end(), less_than_key());
  TLorentzVector *tvec1 = new TLorentzVector();
  TLorentzVector *tvec2 = new TLorentzVector();
  TLorentzVector *tvec3 = new TLorentzVector();
  tvec1->SetPtEtaPhiM(in_jets[0].obj.pt(), in_jets[0].obj.eta(), in_jets[0].obj.phi(), in_jets[0].obj.M());
  tvec2->SetPtEtaPhiM(in_jets[1].obj.pt(), in_jets[1].obj.eta(), in_jets[1].obj.phi(), in_jets[1].obj.M());
  tvec3->SetPtEtaPhiM(in_jets[2].obj.pt(), in_jets[2].obj.eta(), in_jets[2].obj.phi(), in_jets[2].obj.M());
  TLorentzVector W = *tvec2 + *tvec3;
  TLorentzVector top = *tvec1 + *tvec2 + *tvec3;
  double top_mass = top.M();
  //double top_ptDR = top.Pt() * getDeltaR(in_jets[0], (in_jets[1].obj+in_jets[2].obj));
  double top_ptDR = 0;
  double W_mass = W.M();
  double dR = getDeltaR(in_jets[1], in_jets[2]);
  double W_ptDR = W.Pt() * dR;
  double soft_drop = min((*tvec2).Pt(), (*tvec3).Pt())/(((*tvec2).Pt() + (*tvec3).Pt()) * dR * dR);
  double q2_ptd = sqrt((*tvec2).Px()*(*tvec2).Px() + (*tvec2).Py()*(*tvec2).Py() + (*tvec2).Pz()*(*tvec2).Pz())/
                  ((*tvec2).Px() + (*tvec2).Py() + (*tvec2).Pz());
  double q3_ptd = sqrt((*tvec3).Px()*(*tvec3).Px() + (*tvec3).Py()*(*tvec3).Py() + (*tvec3).Pz()*(*tvec3).Pz())/
                  ((*tvec3).Px() + (*tvec3).Py() + (*tvec3).Pz());
  double b_q2_m = (*tvec1 + *tvec2).M();
  double b_q3_m = (*tvec1 + *tvec3).M();

  for (const auto & jet : in_jets)
    {
      in_file<< jet.obj.pt() << "," << jet.obj.eta() <<","<< jet.obj.phi() <<","<< jet.obj.M() <<",";
      in_file<< jet.charge <<","<< jet.DeepCSVprobb <<","<< jet.DeepCSVprobbb <<",";
      in_file<< jet.DeepCSVprobc <<","<< jet.DeepCSVprobudsg <<",";
      in_file<< jet.qgid <<",";
      in_file<< jet.ptD <<","<< jet.axis1 <<","<< jet.mult <<",";
    }
  in_file<< top_mass <<","<< top.Pt() <<","<< top_ptDR <<","<< W_mass <<","<< W_ptDR <<","<< soft_drop;
  in_file<< ","<< q2_ptd <<","<< q3_ptd <<","<< b_q2_m <<","<< b_q3_m;
  
  in_file<<"\n";
  in_file.flush();
}

void run_it(TChain* tree, TString output_file, TString sorted_file, TString bkgd_file, bool signal, bool background, bool selection)
{

  //int num_hadronic = 0;
  int treeentries = tree->GetEntries();   
  cout << "# events in tree: "<< treeentries << endl;  

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

  TFile *outfile = new TFile(output_file, "RECREATE"); //"UPDATE"); // #, 'test' ) // "RECREATE");
  outfile->cd();
  
  ofstream signal_sorted_file;
  ofstream bkgd_sorted_file;

  signal_sorted_file.open(sorted_file);
  bkgd_sorted_file.open(bkgd_file);

  Int_t tree_class = 0;
  ttH::Jet tree_jet1;
  ttH::Jet tree_jet2;
  ttH::Jet tree_jet3;
  TTree tripletree("tripleTree", "Tree of triples of jets");
  tripletree.Branch("class", &tree_class);
  tripletree.Branch("jet1", &tree_jet1);
  tripletree.Branch("jet2", &tree_jet2);
  tripletree.Branch("jet3", &tree_jet3);
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
      //if (!signal && preselected_jets_intree->size() > 5) continue;
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
        } else {
          unmatched_jets.push_back(pjet);
        }
        ix++;
      }
      
      for (const auto &gjet : *gen_parts) {
        int ID = gjet.pdgID;
        bool mother = (gjet.mother != 9999);
        bool grandmother = (gjet.grandmother != 9999);
        int mID = mother ? (*gen_parts)[gjet.mother].pdgID : 0;
        int gmID = grandmother ? (*gen_parts)[gjet.grandmother].pdgID : 0;
        if (abs(ID) == 15) {
          tau_lept = true;
        } else if (ID >= 11 && ID <= 16 && (mID == 24 || gmID == 24)) {
          num_pos_lept++;
        } else if (ID <= -11 && ID >= -16 && (mID == -24 || gmID == -24)) {
          num_neg_lept++;
        }
      }
      
      int size = preselected_jets_intree->size();
      if (selection) {
          // Make sure there's only one hadronic top
          bool matched1 = (matched_jets[0].size() == 3);
          bool matched2 = (matched_jets[1].size() == 3);
          if ((!matched1 == !matched2) || ((num_pos_lept < 2) == (num_neg_lept < 2)) || !tau_lept)
            continue;
      }
      
      //write csv files
      if (signal) {
        tree_class = 1;
        if (matched_jets[0].size() == 3) {
          num_signal++;
          write_csv(signal_sorted_file, matched_jets[0], eventnum_intree, 1);
          tree_jet1 = matched_jets[0][0];
          tree_jet2 = matched_jets[0][1];
          tree_jet3 = matched_jets[0][2];
          tripletree.Fill();
        }
        if (matched_jets[1].size() == 3) {
          num_signal++;
          write_csv(signal_sorted_file, matched_jets[1], eventnum_intree, 1);
          tree_jet1 = matched_jets[1][0];
          tree_jet2 = matched_jets[1][1];
          tree_jet3 = matched_jets[1][2];
          tripletree.Fill();
        }
      } 
      if (background) {
         tree_class = 0;
         int this_bkgd = 0;
         // Generate all combinations of indices via 3 for loops
         // Every iteration make a vector to compare against matched vectors if there is a fully matched triplet
         // Write the triplet to file
         bool matched1 = (matched_jets[0].size() == 3);
         bool matched2 = (matched_jets[1].size() == 3);
         for (int i=0; i < size-2; i++) {
           for (int j=i+1; j < size-1; j++) {
             for (int k=j+1; k < size; k++) {
               vector<int> comb = {i, j, k};
               if (matched1 && (comb == matched_ix[0])) continue;
               if (matched2 && (comb == matched_ix[1])) continue;
               vector<ttH::Jet> bkgd_jets;
               for (const auto& index : comb)
                 bkgd_jets.push_back((*preselected_jets_intree)[index]);
               num_bkgd++;
               this_bkgd++;
               write_csv(bkgd_sorted_file, bkgd_jets, eventnum_intree, 0);
               tree_jet1 = bkgd_jets[0];
               tree_jet2 = bkgd_jets[1];
               tree_jet3 = bkgd_jets[2];
               tripletree.Fill();
             }
           }
         }
         avg_bkgd = (avg_bkgd + this_bkgd)/2;
      }
    }

  double endtime = get_wall_time();
  cout << "Elapsed time: " << endtime - starttime << " seconds, " << endl;
  if (treeentries>0) cout << "an average of " << (endtime - starttime) / treeentries << " per event." << endl;
  cout << "Num Sig: " <<num_signal<< endl;
  cout << "Num Bkgd: " <<num_bkgd<< endl;
  cout << "Avg Bkgd: " <<avg_bkgd<< endl;
  
  tripletree.Write();
  outfile->Close();
  signal_sorted_file.close();
  bkgd_sorted_file.close();
}

void makeFlatTreesForMatt(TString sample="")
{

  TString output_dir = "";
  sample = "ttH";
  TString output_file = output_dir + sample + "_DeepLearningTree" + ".root";
  TChain *tth_chain = new TChain("OSTwoLepAna/summaryTree");
    
  DIR *dir;
  struct dirent *ent;
  dir = opendir ("/scratch365/mdrnevic/trees");
  while ((ent = readdir (dir)) != NULL) {
      tth_chain->Add("/scratch365/mdrnevic/trees/" + (TString) ent->d_name);
  }
  closedir (dir);

  TString sorted_file_csv = output_dir + sample + "_Signal_DeepLearningTree.csv";
  TString bkgd_file_csv = output_dir + sample + "_Background_DeepLearningTree.csv";
    
  bool signal = true;
  bool bkgd = true;
  bool selection = false;
  
  run_it(tth_chain, output_file, sorted_file_csv, bkgd_file_csv, signal, bkgd, selection);
}
