module DFA where

import Prelude hiding (Word)

type State = Int
type Alphabet a = [a]
type DFA a = 
  ( Alphabet a             -- alphabet
  , State                  -- initial state
  , State -> a -> State    -- transition function
  , State -> Bool)         -- test for final state
type Word a = [a]



alphabet :: DFA a -> Alphabet a
alphabet (a, _, _, _) = a

initial :: DFA a -> State
initial (_, a, _, _) = a

transition :: DFA a -> (State -> a -> State)
transition (_, _, a, _)  = a

finalState :: DFA a -> State -> Bool
finalState (_, _, _, a) = a

{-
   Please shortly indicate why using accessor functions is useful.
   Initializing a 'data' structure would have been handier in this case, but I don't know about any other downsides of object types
   
-}

accepts :: DFA a -> Word a -> Bool
accepts dfa inp = finalState' (foldl transition' initial' inp)
    where
        finalState' = finalState dfa
        transition' = transition dfa
        initial' = initial dfa

lexicon :: Alphabet a -> Int -> [Word a]
lexicon alph len = [ x | x<-perm , (length x) == len]
    where
        perm = [ (w1 ++ w2 ++ w3) | w1 <- alph, w2 <- alph, w3 <- alph]

foldl gen

--for each elementin genPerm, call genPerm

genPermMap xs = map genPermRec list --list is list except a beginning character

genPermRec :: Char -> [Char] -> [Char]
genPermRec single [] = single
genPermRec single multi = single + foldr ((++) . genPermMap) single multi --single must be already given --last single and multi should be mutually exclusive


perms :: Char -> [Char] -> [Char]
perms '' [] = ''
perms sgl xs = perms (fst tpl) (snd tpl)
perms '' xs = perms (fst tpl) (snd tpl)
    where 
        tpl = (selectSingle xs)

selectSingle :: Int -> [a] -> (a, as)
selectSingle n xs = (drop (n-1) (take n xs), (take (n-1) xs) ++ (drop n xs) )

{-     
findPerm :: Alphabet a -> Int -> [Word a]
findPerm alph left
    | left == 0 = alph
    | otherwise = map findPerm 
-}    

language :: DFA a -> Int -> [Word a]
language = undefined

-- Try to use map, foldl, foldr, filter and/or list comprehensions.