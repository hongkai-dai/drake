#ifndef __DrakeCollisionBody_H__
#define __DrakeCollisionBody_H__

#include "DrakeCollision.h"

namespace DrakeCollision
{
  template<typename ElementT>
  class Body {
    public:
      Body() : body_idx(-1), parent_idx(-1), group(DEFAULT_GROUP), 
      mask(ALL_MASK), elements() {};

      void addElement(const int body_idx, const int parent_idx, 
                      const Matrix4d& T_elem_to_link, Shape shape, 
                      const std::vector<double>& params)
      {
        //DEBUG
        //std::cout << "Body::addElement: START" << std::endl;
        //END_DEBUG
        if ((this->body_idx == body_idx) && (this->parent_idx == parent_idx)) {
          //DEBUG
          //std::cout << "Body::addElement: Add new element" << std::endl;
          //END_DEBUG
            elements.emplace_back(T_elem_to_link, shape, params);
        } else {
          //DEBUG
          //std::cout << "Body::addElement: Indices don't match" << std::endl;
          //END_DEBUG
          if ((this->body_idx < 0) && (this->parent_idx < 0)) {
            //DEBUG
            //std::cout << "Body::addElement: First element - set indices" << std::endl;
            //END_DEBUG
            this->body_idx = body_idx;
            this->parent_idx = parent_idx;
            elements.emplace_back(T_elem_to_link, shape, params);
            //DEBUG
            //std::cout << "body_idx = " << body_idx << std::endl;
            //std::cout << "this->body_idx = " << this->body_idx << std::endl;
            //END_DEBUG
          } else {
            std::string msg ("DrakeCollision::Body::addElement:");
            if (this->body_idx != body_idx) {
              msg += "Tried to add element with body_idx ";
              msg += std::to_string(body_idx);
              msg += " to a Body with body_idx ";
              msg += std::to_string(this->body_idx);
            } else {
              msg += "Tried to add element with parent_idx ";
              msg += std::to_string(parent_idx);
              msg += " to a Parent with parent_idx ";
              msg += std::to_string(this->parent_idx);
            }
            throw std::invalid_argument(msg);
          }
          //DEBUG
          //std::cout << "Body::addElement: END" << std::endl;
          //END_DEBUG
        }
      };

      const ElementT& operator[] (const int elem_idx) const 
      { 
        return elements[elem_idx];
      };

      const ElementT& at(const int elem_idx) const 
      { 
        return elements.at(elem_idx); 
      };

      const ElementT& back() const { return elements.back(); };

      int getBodyIdx() const { return body_idx; };

      int getParentIdx() const { return parent_idx; };

      const bitmask& getGroup() const { return group; };

      void setGroup(const bitmask& group) { this->group = group; };

      const bitmask& getMask() const { return mask; };

      void setMask(const bitmask& mask) { this->mask = mask; };

      const std::vector<ElementT>& getElements() const { return elements; };

      void addToGroup(const bitmask& group) { this->group |= group; }; 

      void ignoreGroup(const bitmask& group) { this->mask &= ~group; };

      void collideWithGroup(const bitmask& group) { this->mask |= group; };

      void updateElements(const Matrix4d& T_link_to_world)
      {
        for (auto elem : elements) {
          elem.updateWorldTransform(T_link_to_world);
        }
      };

      bool adjacentTo(const Body<ElementT>& other) const
      {
        return (body_idx == other.getParentIdx()) || 
               (parent_idx == other.getBodyIdx());
      };

      bool collidesWith(const Body<ElementT>& other) const
      {
        return (group & other.mask).any() && (other.group & mask).any();
      };

    private:
      int body_idx;
      int parent_idx;
      bitmask group;
      bitmask  mask;
      std::vector<ElementT> elements;
  };
}
#endif

