---
layout:     post
title:      Git Best Practices
subtitle:   
date:       2019-01-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-rwd.jpg"
catalog: true
tags:
    - git
---


# Git Best Practices



![Git Data Transport Commands](https://www.patrickzahnd.ch/uploads/git-transport-v1-1024x723.png)

`git pull`  =   `fetch`  +   `merge`. 

`git pull --rebase`  =  `fetch`   +    `rebase`



![Git's file system (consisting of the working directory, staging area and HEAD as well as a series of commands to move files between the three file systems)](https://mukulrathi.com/static/bcdea94954bfedcb356de751d5cc010f/e2622/git-file-system.png)



##### Config
```bash
git config color.ui true 

git config format.pretty oneline 
git config log.abbrevCommit true
```



##### Add

`git add -i`  	use interactive adding

`git add .`  	add all files

##### Unstage

`git rm --cached <file>`   unstage a staged new file

`git reset <file>`    unstage a staged modified file ,  same as `git reset HEAD <file>`;

mechanism: make use of  `git reset <target_commit>` , resets the index but not the working tree

```bash
working index HEAD target         working index HEAD
----------------------------------------------------
 A       B     C    C     --soft   A       B     C
                          --mixed  A       C     C  # default
                          --hard   C       C     C
```

##### Undo working

`git checkout <file>`   overwrite  working tree from index or from commit(HEAD). 
if (index has file) {always from index} else {from HEAD}

`git checkout HEAD <file>`   update working and index from commit(HEAD)



##### Uncommit

`git reset HEAD~1  `  undo previous commit

```bash
working index HEAD target         working index HEAD
----------------------------------------------------
 A       B     C    D     --soft   A       B     D
                          --mixed  A       D     D
                          --hard   D       D     D
```



##### Recover missing commit

```bash
git reflog
```



##### Github amend commit

 ```shell
git commit --amend
git push -f  # --force , discourage
 ```



##### 3-Way Merge Flow

![imgae](https://www.ntu.edu.sg/home/ehchua/programming/howto/images/Git_Merge3Way.png)



```bash
$ git checkout master
// undo the Commit-4, back to Commit-3
$ git reset --hard HEAD~1
HEAD is now at 7e7cb40 Commit 3

// Change the email to abc@abc.com
$ git add README.md
$ git commit -m "Commit 5"

$ git checkout devel
// undo the Commit-4, back to Commit-3
$ git reset --hard HEAD~1
// Change the email to xyz@xyz.com to trigger conflict
$ git add README.md
$ git commit -m "Commit 4"

# do a 3-way merge with conflict
$ git checkout master
$ git merge devel
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
Automatic merge failed; fix conflicts and then commit the result.
 
$ git status
```

The conflict file is marked as follows (in "`git status`"):

```bash
<<<<<<< HEAD
This is the README. My email is abc@abc.com
=======
This is the README. My email is xyz@xyz.com
>>>>>>> devel
This line is added after Commit 1
This line is added after Commit 2
```

You need to manually decide which way to take, or you could discard both by setting the email to zzz@nowhere.com.

```
$ git add README.md
$ git commit -m "Commit 6"
```



#####  Rebase

![imgae](https://www.ntu.edu.sg/home/ehchua/programming/howto/images/Git_Rebase1.png)

```shell
// Start a new feature branch from the current master
$ git checkout -b feature master
// Edit/Stage/Commit changes to feature branch
 
// Need to work on a fix on the master
$ git checkout -b hotfix master
// Edit/Stage/Commit changes to hotfix branch
// Merge hotfix into master
$ git checkout master
$ git merge hotfix
// Delete hotfix branch
$ git branch -d hotfix
 
// Rebase feature branch on master branch
//  to maintain a linear history
$ git checkout feature
$ git rebase master
// Now, linear merge
$ git checkout master
$ git merge feature
```



##### Forking Workflow

![Forking Workflow](https://www.ntu.edu.sg/home/ehchua/programming/howto/images/Git_ForkingWorkFlow.png)



##### Git Flow

![img](https://www.patrickzahnd.ch/uploads/gitflow.png)



### References

How to get started with GIT https://www.ntu.edu.sg/home/ehchua/programming/howto/Git_HowTo.html

git and svn data transport commands https://www.patrickzahnd.ch/blog.html

The Ultimate Git Beginner Reference Guide https://mukulrathi.com/git-beginner-cheatsheet/